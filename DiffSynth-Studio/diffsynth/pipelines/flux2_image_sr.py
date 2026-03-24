import os
import torch, math, torchvision
from PIL import Image
from typing import Union
from tqdm import tqdm
from einops import rearrange
import numpy as np
from typing import Union, List, Optional, Tuple

from ..core.device.npu_compatible_device import get_device_type
from ..diffusion import FlowMatchScheduler
from ..core import ModelConfig, gradient_checkpoint_forward
from ..diffusion.base_pipeline import BasePipeline, PipelineUnit, ControlNetInput

from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration
from transformers.utils import logging as transformers_logging
from ..models.flux2_dit import Flux2DiT
from ..models.flux2_vae import Flux2VAE


QWEN35_IMAGE_SR_SYSTEM_PROMPT = (
    "You are an AI that reasons about image descriptions. Provide structured responses focused on object relationships, "
    "object attribution, and actions without speculation. Prioritize reading visible text in the image, including exact "
    "characters, words, phrases, line breaks, layout, and local text regions."
)


def _imagesr_debug_enabled() -> bool:
    return os.environ.get("IMAGESR_DEBUG", "0") == "1"


def _imagesr_rank() -> str:
    return os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))


def _imagesr_log(message: str, debug_only: bool = False, once_key: Optional[str] = None):
    if debug_only and not _imagesr_debug_enabled():
        return

    rank = _imagesr_rank()
    if not debug_only and rank not in ("0", "-1"):
        return

    if once_key is not None:
        if not hasattr(_imagesr_log, "_once_keys"):
            _imagesr_log._once_keys = set()
        if once_key in _imagesr_log._once_keys:
            return
        _imagesr_log._once_keys.add(once_key)

    print(f"[ImageSR][Qwen35][rank={rank}] {message}", flush=True)


def _apply_mistral_regex_fix(tokenizer_or_processor, model_path: str):
    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    patch_fn = getattr(type(tokenizer), "_patch_mistral_regex", None)
    if patch_fn is None:
        return tokenizer_or_processor

    try:
        patched_tokenizer = patch_fn(
            tokenizer,
            model_path,
            init_kwargs={"fix_mistral_regex": True},
            fix_mistral_regex=True,
            local_files_only=True,
            is_local=True,
        )
        setattr(patched_tokenizer, "fix_mistral_regex", True)
        if hasattr(tokenizer_or_processor, "tokenizer"):
            tokenizer_or_processor.tokenizer = patched_tokenizer
        else:
            tokenizer_or_processor = patched_tokenizer
        _imagesr_log(f"Applied Mistral regex fix for tokenizer from: {model_path}", debug_only=True, once_key=f"regex_fix:{model_path}")
    except Exception:
        pass
    return tokenizer_or_processor


def _load_pretrained_quietly(load_fn, model_path: str):
    previous_verbosity = transformers_logging.get_verbosity()
    transformers_logging.set_verbosity_error()
    try:
        loaded = load_fn(model_path)
    finally:
        transformers_logging.set_verbosity(previous_verbosity)
    return loaded


class Qwen35PromptAligner(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_groups: int = 8, rank: Optional[int] = 256):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_groups = num_groups
        self.rank = rank
        self.use_grouped_low_rank = (
            rank > 0 and in_dim % num_groups == 0 and out_dim % num_groups == 0
        )

        if self.use_grouped_low_rank:
            self.in_dim_per_group = in_dim // num_groups
            self.out_dim_per_group = out_dim // num_groups
            self.layer_mix = torch.nn.Parameter(torch.eye(num_groups))
            self.input_norms = torch.nn.ModuleList(
                [torch.nn.LayerNorm(self.in_dim_per_group) for _ in range(num_groups)]
            )
            self.down_projs = torch.nn.ModuleList(
                [torch.nn.Linear(self.in_dim_per_group, rank, bias=False) for _ in range(num_groups)]
            )
            self.up_projs = torch.nn.ModuleList(
                [torch.nn.Linear(rank, self.out_dim_per_group, bias=False) for _ in range(num_groups)]
            )
            self.act = torch.nn.SiLU()

            for down_proj, up_proj in zip(self.down_projs, self.up_projs):
                torch.nn.init.xavier_uniform_(down_proj.weight)
                torch.nn.init.xavier_uniform_(up_proj.weight)
        else:
            self.proj = torch.nn.Linear(in_dim, out_dim, bias=False)
            if in_dim == out_dim:
                with torch.no_grad():
                    self.proj.weight.zero_()
                    eye_size = min(in_dim, out_dim)
                    self.proj.weight[:eye_size, :eye_size] = torch.eye(eye_size, dtype=self.proj.weight.dtype)
            else:
                torch.nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x):
        if not self.use_grouped_low_rank:
            return self.proj(x)

        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_groups, self.in_dim_per_group)
        x = torch.einsum("blgi,gh->blhi", x, self.layer_mix)

        aligned_groups = []
        for group_id in range(self.num_groups):
            hidden = self.input_norms[group_id](x[:, :, group_id, :])
            hidden = self.down_projs[group_id](hidden)
            hidden = self.act(hidden)
            hidden = self.up_projs[group_id](hidden)
            aligned_groups.append(hidden)

        return torch.cat(aligned_groups, dim=-1)

    @property
    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters())

    def _reference_parameter(self):
        return next(self.parameters())


class Flux2ImagePipeline(BasePipeline):

    def __init__(self, device=get_device_type(), torch_dtype=torch.bfloat16):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16,
        )
        self.scheduler = FlowMatchScheduler("FLUX.2")
        # self.text_encoder: Flux2TextEncoder = None  # legacy path, disabled
        # self.text_encoder_qwen3: ZImageTextEncoder = None  # legacy path, disabled
        self.text_encoder_qwen35 = None
        self.dit: Flux2DiT = None
        self.vae: Flux2VAE = None
        self.tokenizer: Union[AutoTokenizer, AutoProcessor] = None
        self.processor_qwen35: AutoProcessor = None
        self.qwen35_prompt_aligner: Qwen35PromptAligner = None
        self.in_iteration_models = ("dit",)
        self.units = [
            Flux2Unit_ShapeChecker(),
            # Flux2Unit_PromptEmbedder(),  # legacy text-only encoder path (disabled for Qwen3.5 training chain)
            Flux2Unit_Qwen35ImageTextEmbedder(),
            Flux2Unit_NoiseInitializer(),
            Flux2Unit_InputImageEmbedder(),
            Flux2Unit_EditImageEmbedder(),
            Flux2Unit_ImageIDs(),
        ]
        self.model_fn = model_fn_flux2
    
    
    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = get_device_type(),
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="black-forest-labs/FLUX.2-dev", origin_file_pattern="tokenizer/"),
        vram_limit: float = None,
        qwen35_processor_config: ModelConfig = None,
    ):
        # Initialize pipeline
        pipe = Flux2ImagePipeline(device=device, torch_dtype=torch_dtype)
        model_pool = pipe.download_and_load_models(model_configs, vram_limit)
        
        # Fetch models
        # pipe.text_encoder = model_pool.fetch_model("flux2_text_encoder")  # legacy path, disabled
        # pipe.text_encoder_qwen3 = model_pool.fetch_model("z_image_text_encoder")  # legacy path, disabled
        pipe.text_encoder_qwen35 = model_pool.fetch_model("qwen35_image_text_encoder")
        pipe.dit = model_pool.fetch_model("flux2_dit")
        pipe.vae = model_pool.fetch_model("flux2_vae")

        # Fallback: load Qwen3.5 image-text encoder directly from Hugging Face / local path.
        if pipe.text_encoder_qwen35 is None and qwen35_processor_config is not None:
            try:
                qwen35_processor_config.download_if_necessary()
                pipe.text_encoder_qwen35 = Qwen3_5ForConditionalGeneration.from_pretrained(
                    qwen35_processor_config.path,
                    torch_dtype=torch_dtype,
                )
            except Exception:
                pipe.text_encoder_qwen35 = None

        if pipe.text_encoder_qwen35 is not None:
            processor_config = qwen35_processor_config if qwen35_processor_config is not None else tokenizer_config
            if processor_config is not None:
                processor_config.download_if_necessary()
                try:
                    pipe.processor_qwen35 = _load_pretrained_quietly(AutoProcessor.from_pretrained, processor_config.path)
                    pipe.processor_qwen35 = _apply_mistral_regex_fix(pipe.processor_qwen35, processor_config.path)
                except Exception:
                    pipe.processor_qwen35 = None
                # Keep a tokenizer from the same Qwen3.5 source as fallback for text-only input.
                if pipe.processor_qwen35 is None:
                    try:
                        pipe.tokenizer = _load_pretrained_quietly(AutoTokenizer.from_pretrained, processor_config.path)
                        pipe.tokenizer = _apply_mistral_regex_fix(pipe.tokenizer, processor_config.path)
                    except Exception:
                        pipe.tokenizer = None
            # Build trainable alignment layer so Qwen3.5 text features match DiT expected context dim.
            if hasattr(pipe.text_encoder_qwen35, "model") and hasattr(pipe.text_encoder_qwen35.model, "config"):
                hidden_size = pipe.text_encoder_qwen35.model.config.text_config.hidden_size
            else:
                hidden_size = pipe.text_encoder_qwen35.config.text_config.hidden_size
            qwen_hidden_states_layers = 3  # Must be consistent with Flux2Unit_Qwen35ImageTextEmbedder.num_hidden_state_layers
            in_dim = hidden_size * qwen_hidden_states_layers
            out_dim = pipe.dit.context_embedder.in_features if pipe.dit is not None else in_dim
            pipe.qwen35_prompt_aligner = Qwen35PromptAligner(in_dim, out_dim).to(device=pipe.device, dtype=pipe.torch_dtype)
            processor_name = type(pipe.processor_qwen35).__name__ if pipe.processor_qwen35 is not None else None
            tokenizer_name = type(pipe.tokenizer).__name__ if pipe.tokenizer is not None else None
            _imagesr_log(
                "Qwen35ImageTextEmbedder loaded successfully "
                f"(text_encoder={type(pipe.text_encoder_qwen35).__name__}, "
                f"processor={processor_name}, tokenizer_fallback={tokenizer_name}, "
                f"aligner={in_dim}->{out_dim}, dtype={pipe.torch_dtype}, device={pipe.device})",
                once_key="qwen35_loaded_success",
            )
            _imagesr_log(
                f"Qwen35 processor source: {processor_config.path if processor_config is not None else 'None'}",
                debug_only=True,
                once_key="qwen35_processor_source",
            )
        elif qwen35_processor_config is not None:
            _imagesr_log(
                "Qwen35ImageTextEmbedder is NOT active because text_encoder_qwen35 was not loaded.",
                once_key="qwen35_loaded_failed",
            )
        # elif tokenizer_config is not None:
        #     # Legacy fallback path when Qwen3.5 is not available.
        #     tokenizer_config.download_if_necessary()
        #     pipe.tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.path)
        
        # VRAM Management
        pipe.vram_management_enabled = pipe.check_vram_management_state()
        return pipe
    
    
    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: str = "",
        cfg_scale: float = 1.0,
        embedded_guidance: float = 4.0,
        # Image
        input_image: Image.Image = None,
        denoising_strength: float = 1.0,
        # Encode image (for Qwen3.5 image-text encoder only)
        encode_image: Union[Image.Image, List[Image.Image]] = None,
        encode_image_auto_resize: bool = True,
        # Edit
        edit_image: Union[Image.Image, List[Image.Image]] = None,
        edit_image_auto_resize: bool = True,
        # Shape
        height: int = 1024,
        width: int = 1024,
        # Randomness
        seed: int = None,
        rand_device: str = "cpu",
        # Steps
        num_inference_steps: int = 30,
        # Progress bar
        progress_bar_cmd = tqdm,
    ):
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, dynamic_shift_len=height//16*width//16)

        # Parameters
        inputs_posi = {
            "prompt": prompt,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
        }
        inputs_shared = {
            "cfg_scale": cfg_scale, "embedded_guidance": embedded_guidance,
            "input_image": input_image, "denoising_strength": denoising_strength,
            "encode_image": encode_image, "encode_image_auto_resize": encode_image_auto_resize,
            "edit_image": edit_image, "edit_image_auto_resize": edit_image_auto_resize,
            "height": height, "width": width,
            "seed": seed, "rand_device": rand_device,
            "num_inference_steps": num_inference_steps,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            noise_pred = self.cfg_guided_model_fn(
                self.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = self.step(self.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs_shared)
        
        # Decode
        self.load_models_to_device(['vae'])
        latents = rearrange(inputs_shared["latents"], "B (H W) C -> B C H W", H=inputs_shared["height"]//16, W=inputs_shared["width"]//16)
        image = self.vae.decode(latents)
        image = self.vae_output_to_image(image)
        self.load_models_to_device([])

        return image


class Flux2Unit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("height", "width"),
        )

    def process(self, pipe: Flux2ImagePipeline, height, width):
        height, width = pipe.check_resize_height_width(height, width)
        return {"height": height, "width": width}


# class Flux2Unit_PromptEmbedder(PipelineUnit):
#     Legacy Mistral/FLUX text-only embedder path.
#     Disabled because this training chain now uses Qwen3.5 image-text embeddings exclusively.
#
# class Flux2Unit_Qwen3PromptEmbedder(PipelineUnit):
#     Legacy Qwen3 text-only embedder path.
#     Disabled because this training chain now uses Qwen3.5 image-text embeddings exclusively.


class Flux2Unit_Qwen35ImageTextEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params=("encode_image", "encode_image_auto_resize"),
            input_params_posi={"prompt": "prompt"},
            input_params_nega={"prompt": "negative_prompt"},
            output_params=("prompt_emb", "prompt_emb_mask"),
            onload_model_names=("text_encoder_qwen35", "qwen35_prompt_aligner"),
        )
        self.num_hidden_state_layers = 3
        self.hidden_state_layer_ratios = (1 / 3, 2 / 3, 1.0)
        self._debug_runtime_calls = 0
        self._debug_embed_calls = 0

    def select_hidden_state_layers(self, hidden_states):
        total_states = len(hidden_states)
        if total_states <= 0:
            raise ValueError("Qwen3.5 output does not contain hidden states.")

        max_index = total_states - 1
        selected = []
        for ratio in self.hidden_state_layer_ratios:
            idx = int(round(max_index * ratio))
            idx = min(max(idx, 0), max_index)
            if idx not in selected:
                selected.append(idx)

        for idx in range(max_index, -1, -1):
            if len(selected) >= self.num_hidden_state_layers:
                break
            if idx not in selected:
                selected.append(idx)

        selected = sorted(selected[: self.num_hidden_state_layers])
        return selected

    def calculate_dimensions(self, target_area, ratio):
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return width, height

    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height * scale), round(width * scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def encode_image_auto_resize(self, encode_image):
        ratio = encode_image.size[0] / encode_image.size[1]
        calculated_width, calculated_height = self.calculate_dimensions(1024 * 1024, ratio)
        return self.crop_and_resize(encode_image, calculated_height, calculated_width)

    def normalize_encode_images(self, encode_image, auto_resize: bool):
        if encode_image is None:
            return None
        if isinstance(encode_image, Image.Image):
            encode_images = [encode_image]
        else:
            encode_images = list(encode_image)
        out_images = []
        for image in encode_images:
            if not isinstance(image, Image.Image):
                raise TypeError("encode_image must be a PIL.Image or a list of PIL.Image.")
            if auto_resize is None or auto_resize:
                image = self.encode_image_auto_resize(image)
            out_images.append(image)
        return out_images

    def _resolve_dtype_device(self, text_encoder, dtype=None, device=None):
        if dtype is None:
            dtype = getattr(text_encoder, "dtype", None)
        if device is None:
            device = getattr(text_encoder, "device", None)
        if dtype is None or device is None:
            try:
                p = next(text_encoder.parameters())
                if dtype is None:
                    dtype = p.dtype
                if device is None:
                    device = p.device
            except StopIteration:
                pass
        if dtype is None:
            dtype = torch.float32
        if device is None:
            device = torch.device("cpu")
        return dtype, device

    def _build_system_prompt(self) -> str:
        return QWEN35_IMAGE_SR_SYSTEM_PROMPT

    def _build_text_messages(self, prompt: str):
        return [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": prompt},
        ]

    def _build_image_text_messages(self, prompt: str, images: List[Image.Image]):
        image_contents = [{"type": "image", "image": image} for image in images]
        return [
            {"role": "system", "content": self._build_system_prompt()},
            {
                "role": "user",
                "content": image_contents + [{"type": "text", "text": prompt}],
            },
        ]

    def _build_manual_chat_text(self, user_content: str):
        system_prompt = self._build_system_prompt()
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
        )

    def _build_text_inputs(self, tokenizer, prompt: str, max_sequence_length: int):
        messages = self._build_text_messages(prompt)
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

    def _build_image_text_inputs(self, processor, prompt: str, images: Optional[List[Image.Image]], max_sequence_length: int):
        if images is None:
            return self._build_text_inputs(processor, prompt, max_sequence_length)

        if isinstance(images, Image.Image):
            images = [images]

        tokenizer = getattr(processor, "tokenizer", processor)
        vision_bos_token = getattr(tokenizer, "vision_bos_token", "<|vision_start|>")
        image_token = getattr(tokenizer, "image_token", "<|image_pad|>")
        vision_eos_token = getattr(tokenizer, "vision_eos_token", "<|vision_end|>")

        # 512 is too short for multimodal prompts after vision token expansion.
        if max_sequence_length is None or max_sequence_length < 2048:
            max_sequence_length = 2048

        def run_processor(text, truncation: bool):
            processor_kwargs = {
                "text": [text],
                "images": images,
                "return_tensors": "pt",
            }
            if truncation:
                processor_kwargs["padding"] = "max_length"
                processor_kwargs["truncation"] = True
                processor_kwargs["max_length"] = max_sequence_length
            else:
                processor_kwargs["padding"] = True
                processor_kwargs["truncation"] = False
            return processor(**processor_kwargs)

        def build_fallback_text():
            image_placeholders = "".join(
                f"{vision_bos_token}{image_token}{vision_eos_token}\n"
                for _ in images
            )
            return self._build_manual_chat_text(image_placeholders + prompt)

        messages = self._build_image_text_messages(prompt, images)

        text = None
        try:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = None

        fallback_used = False
        if text is None or image_token not in text:
            text = build_fallback_text()
            fallback_used = True

        try:
            inputs = run_processor(text, truncation=True)
        except ValueError as error:
            if "Mismatch in `image` token count" in str(error):
                inputs = run_processor(text, truncation=False)
                _imagesr_log(
                    f"Retry without truncation due image token mismatch. prompt_len={len(prompt)}, num_images={len(images)}",
                    debug_only=True,
                )
            else:
                raise
        input_ids = inputs.get("input_ids")
        image_token_id = getattr(tokenizer, "image_token_id", None)

        if image_token_id is not None and torch.is_tensor(input_ids):
            image_token_count = int((input_ids == image_token_id).sum().item())
            if image_token_count == 0:
                fallback_text = build_fallback_text()
                try:
                    inputs = run_processor(fallback_text, truncation=True)
                except ValueError as error:
                    if "Mismatch in `image` token count" in str(error):
                        inputs = run_processor(fallback_text, truncation=False)
                    else:
                        raise
                fallback_used = True
                input_ids = inputs.get("input_ids")
                image_token_count = int((input_ids == image_token_id).sum().item()) if torch.is_tensor(input_ids) else -1
            _imagesr_log(
                f"_build_image_text_inputs: fallback_used={fallback_used}, image_token_count={image_token_count}, "
                f"input_ids_shape={tuple(input_ids.shape) if torch.is_tensor(input_ids) else None}",
                debug_only=True,
            )
        else:
            _imagesr_log(
                f"_build_image_text_inputs: fallback_used={fallback_used}, image_token_id unavailable.",
                debug_only=True,
            )

        return inputs

    def get_qwen35_prompt_embeds(
        self,
        text_encoder,
        tokenizer: Union[AutoProcessor, AutoTokenizer],
        prompt: Union[str, List[str]],
        encode_images: Optional[List[Image.Image]],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        max_sequence_length: int = 2048,
    ):
        dtype, device = self._resolve_dtype_device(text_encoder, dtype=dtype, device=device)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        image_groups = [None] * len(prompt)
        if encode_images is not None:
            if len(prompt) == 1:
                image_groups = [encode_images]
            elif len(encode_images) == 1:
                image_groups = [[encode_images[0]] for _ in prompt]
            elif len(encode_images) == len(prompt):
                image_groups = [[image] for image in encode_images]
            else:
                raise ValueError("encode_image length must be 1 or equal to prompt length when using prompt list.")

        batched_inputs = {}
        for single_prompt, single_images in zip(prompt, image_groups):
            try:
                inputs = self._build_image_text_inputs(tokenizer, single_prompt, single_images, max_sequence_length)
            except Exception:
                inputs = self._build_text_inputs(tokenizer, single_prompt, max_sequence_length)

            for key, value in inputs.items():
                if torch.is_tensor(value):
                    batched_inputs.setdefault(key, []).append(value)

        batched_inputs = {
            key: torch.cat(values, dim=0).to(device)
            for key, values in batched_inputs.items()
        }

        output = text_encoder(
            **batched_inputs,
            output_hidden_states=True,
            use_cache=False,
        )

        hidden_states = output.hidden_states
        layers = self.select_hidden_state_layers(hidden_states)
        out = torch.stack([hidden_states[i] for i in layers], dim=1)
        out = out.to(dtype=dtype, device=device)

        bsz, channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(bsz, seq_len, channels * hidden_dim)
        self._debug_embed_calls += 1
        if self._debug_embed_calls <= 5 or self._debug_embed_calls % 50 == 0:
            _imagesr_log(
                f"get_qwen35_prompt_embeds call={self._debug_embed_calls}: "
                f"prompt_embeds_shape={tuple(prompt_embeds.shape)}, dtype={prompt_embeds.dtype}, device={prompt_embeds.device}",
                debug_only=True,
            )
        return prompt_embeds

    def prepare_text_ids(
        self,
        x: torch.Tensor,  # (B, L, D)
        t_coord: Optional[torch.Tensor] = None,
    ):
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)
            out_ids.append(torch.cartesian_prod(t, h, w, l))

        return torch.stack(out_ids)

    def encode_prompt(
        self,
        text_encoder,
        tokenizer: Union[AutoProcessor, AutoTokenizer],
        prompt: Union[str, List[str]],
        encode_image: Union[Image.Image, List[Image.Image], None],
        encode_image_auto_resize: bool = True,
        dtype=None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 2048,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt

        encode_images = self.normalize_encode_images(encode_image, encode_image_auto_resize)

        if prompt_embeds is None:
            prompt_embeds = self.get_qwen35_prompt_embeds(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompt,
                encode_images=encode_images,
                dtype=dtype,
                device=device,
                max_sequence_length=max_sequence_length,
            )

        batch_size, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        text_ids = self.prepare_text_ids(prompt_embeds).to(device)
        return prompt_embeds, text_ids

    def process(self, pipe: Flux2ImagePipeline, prompt, encode_image, encode_image_auto_resize):
        text_encoder_qwen35 = getattr(pipe, "text_encoder_qwen35", None)
        if text_encoder_qwen35 is None:
            return {}

        processor = getattr(pipe, "processor_qwen35", None)
        if processor is None:
            processor = pipe.tokenizer

        if encode_image is not None and not hasattr(processor, "image_processor"):
            raise ValueError(
                "Qwen3.5 image-text encoding requires an AutoProcessor with image_processor. "
                "Please pass qwen35_processor_config to Flux2ImagePipeline.from_pretrained()."
            )
        self._debug_runtime_calls += 1
        if self._debug_runtime_calls <= 5 or self._debug_runtime_calls % 50 == 0:
            _imagesr_log(
                f"Qwen35ImageTextEmbedder runtime call={self._debug_runtime_calls}: "
                f"encode_image={'yes' if encode_image is not None else 'no'}, "
                f"auto_resize={encode_image_auto_resize}, prompt_type={type(prompt).__name__}",
                debug_only=True,
            )
            _imagesr_log(
                f"System prompt active: {self._build_system_prompt()}",
                debug_only=True,
                once_key="qwen35_system_prompt_active",
            )

        pipe.load_models_to_device(self.onload_model_names)
        prompt_embeds, text_ids = self.encode_prompt(
            text_encoder=text_encoder_qwen35,
            tokenizer=processor,
            prompt=prompt,
            encode_image=encode_image,
            encode_image_auto_resize=encode_image_auto_resize,
            dtype=pipe.torch_dtype,
            device=pipe.device,
        )

        aligner = getattr(pipe, "qwen35_prompt_aligner", None)
        if aligner is not None:
            ref_param = aligner._reference_parameter()
            if prompt_embeds.device != ref_param.device:
                prompt_embeds = prompt_embeds.to(device=ref_param.device)
            if prompt_embeds.dtype != ref_param.dtype:
                prompt_embeds = prompt_embeds.to(dtype=ref_param.dtype)
            prompt_embeds = aligner(prompt_embeds)
        if self._debug_runtime_calls <= 5 or self._debug_runtime_calls % 50 == 0:
            _imagesr_log(
                f"Qwen35ImageTextEmbedder output call={self._debug_runtime_calls}: "
                f"prompt_embeds_shape={tuple(prompt_embeds.shape)}, text_ids_shape={tuple(text_ids.shape)}, "
                f"aligner={'on' if aligner is not None else 'off'}",
                debug_only=True,
            )

        return {"prompt_embeds": prompt_embeds, "text_ids": text_ids}


class Flux2Unit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "seed", "rand_device"),
            output_params=("noise",),
        )

    def process(self, pipe: Flux2ImagePipeline, height, width, seed, rand_device):
        noise = pipe.generate_noise((1, 128, height//16, width//16), seed=seed, rand_device=rand_device, rand_torch_dtype=pipe.torch_dtype)
        noise = noise.reshape(1, 128, height//16 * width//16).permute(0, 2, 1)
        return {"noise": noise}


class Flux2Unit_InputImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "noise"),
            output_params=("latents", "input_latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: Flux2ImagePipeline, input_image, noise):
        if input_image is None:
            return {"latents": noise, "input_latents": None}
        pipe.load_models_to_device(['vae'])
        image = pipe.preprocess_image(input_image)
        input_latents = pipe.vae.encode(image)
        input_latents = rearrange(input_latents, "B C H W -> B (H W) C")
        if pipe.scheduler.training:
            return {"latents": noise, "input_latents": input_latents}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents, "input_latents": input_latents}


class Flux2Unit_EditImageEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("edit_image", "edit_image_auto_resize"),
            output_params=("edit_latents", "edit_image_ids"),
            onload_model_names=("vae",)
        )

    def calculate_dimensions(self, target_area, ratio):
        import math
        width = math.sqrt(target_area * ratio)
        height = width / ratio
        width = round(width / 32) * 32
        height = round(height / 32) * 32
        return width, height
    
    def crop_and_resize(self, image, target_height, target_width):
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        image = torchvision.transforms.functional.center_crop(image, (target_height, target_width))
        return image

    def edit_image_auto_resize(self, edit_image):
        calculated_width, calculated_height = self.calculate_dimensions(1024 * 1024, edit_image.size[0] / edit_image.size[1])
        return self.crop_and_resize(edit_image, calculated_height, calculated_width)
    
    def process_image_ids(self, image_latents, scale=10):
        t_coords = [scale + scale * t for t in torch.arange(0, len(image_latents))]
        t_coords = [t.view(-1) for t in t_coords]

        image_latent_ids = []
        for x, t in zip(image_latents, t_coords):
            x = x.squeeze(0)
            _, height, width = x.shape

            x_ids = torch.cartesian_prod(t, torch.arange(height), torch.arange(width), torch.arange(1))
            image_latent_ids.append(x_ids)

        image_latent_ids = torch.cat(image_latent_ids, dim=0)
        image_latent_ids = image_latent_ids.unsqueeze(0)

        return image_latent_ids

    def process(self, pipe: Flux2ImagePipeline, edit_image, edit_image_auto_resize):
        if edit_image is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        if isinstance(edit_image, Image.Image):
            edit_image = [edit_image]
        resized_edit_image, edit_latents = [], []
        for image in edit_image:
            # Preprocess
            if edit_image_auto_resize is None or edit_image_auto_resize:
                image = self.edit_image_auto_resize(image)
            resized_edit_image.append(image)
            # Encode
            image = pipe.preprocess_image(image)
            latents = pipe.vae.encode(image)
            edit_latents.append(latents)
        edit_image_ids = self.process_image_ids(edit_latents).to(pipe.device)
        edit_latents = torch.concat([rearrange(latents, "B C H W -> B (H W) C") for latents in edit_latents], dim=1)
        return {"edit_latents": edit_latents, "edit_image_ids": edit_image_ids}


class Flux2Unit_ImageIDs(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width"),
            output_params=("image_ids",),
        )

    def prepare_latent_ids(self, height, width):
        t = torch.arange(1)  # [0] - time dimension
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)  # [0] - layer dimension

        # Create position IDs: (H*W, 4)
        latent_ids = torch.cartesian_prod(t, h, w, l)

        # Expand to batch: (B, H*W, 4)
        latent_ids = latent_ids.unsqueeze(0).expand(1, -1, -1)

        return latent_ids

    def process(self, pipe: Flux2ImagePipeline, height, width):
        image_ids = self.prepare_latent_ids(height // 16, width // 16).to(pipe.device)
        return {"image_ids": image_ids}


def model_fn_flux2(
    dit: Flux2DiT,
    latents=None,
    timestep=None,
    embedded_guidance=None,
    prompt_embeds=None,
    text_ids=None,
    image_ids=None,
    edit_latents=None,
    edit_image_ids=None,
    use_gradient_checkpointing=False,
    use_gradient_checkpointing_offload=False,
    **kwargs,
):
    image_seq_len = latents.shape[1]
    if edit_latents is not None:
        image_seq_len = latents.shape[1]
        latents = torch.concat([latents, edit_latents], dim=1)
        image_ids = torch.concat([image_ids, edit_image_ids], dim=1)
    embedded_guidance = torch.tensor([embedded_guidance], device=latents.device)
    model_output = dit(
        hidden_states=latents,
        timestep=timestep / 1000,
        guidance=embedded_guidance,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=image_ids,
        use_gradient_checkpointing=use_gradient_checkpointing,
        use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
    )
    model_output = model_output[:, :image_seq_len]
    return model_output
