import argparse
import json
import os
import shutil
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

from diffsynth.core import ModelConfig, load_state_dict
from diffsynth.pipelines.flux2_image_sr import Flux2ImagePipeline
from diffsynth.pipelines import flux2_image_sr as flux2_image_sr_module

os.environ["TOKENIZERS_PARALLELISM"] = "false"


DEFAULT_PROMPT = (
    "Restore and enhance the low-quality input image to generate a high-resolution, "
    "high-quality version. The text within the image should be clear, accurate, and unchanged."
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_path_or_model_id(path_or_model_id: Optional[str]) -> Optional[ModelConfig]:
    if path_or_model_id is None:
        return None
    if os.path.exists(path_or_model_id):
        return ModelConfig(path=path_or_model_id)
    if ":" not in path_or_model_id:
        return ModelConfig(model_id=path_or_model_id, origin_file_pattern=None)
    split_id = path_or_model_id.rfind(":")
    model_id = path_or_model_id[:split_id]
    origin_file_pattern = path_or_model_id[split_id + 1 :]
    return ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern)


def split_model_specs(models: Optional[str]) -> List[str]:
    if models is None:
        return []
    return [x for x in models.split(",") if x]


def is_vae_model_spec(spec: str) -> bool:
    parsed = parse_path_or_model_id(spec)
    candidates = [
        spec,
        None if parsed is None else parsed.path,
        None if parsed is None else parsed.model_id,
        None if parsed is None else parsed.origin_file_pattern,
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        normalized = str(candidate).replace("\\", "/").lower()
        if "/vae/" in normalized or normalized.startswith("vae/") or normalized.endswith("/vae"):
            return True
        if "flux2_vae" in normalized:
            return True
    return False


def disable_vae_fp8_for_inference(fp8_models: Optional[str]) -> Optional[str]:
    specs = split_model_specs(fp8_models)
    if not specs:
        return fp8_models
    kept_specs = [spec for spec in specs if not is_vae_model_spec(spec)]
    if len(kept_specs) == len(specs):
        return fp8_models
    return None if len(kept_specs) == 0 else ",".join(kept_specs)


def parse_vram_config(fp8: bool = False, offload: bool = False, device: str = "cpu") -> Dict:
    if fp8:
        return {
            "offload_dtype": torch.float8_e4m3fn,
            "offload_device": device,
            "onload_dtype": torch.float8_e4m3fn,
            "onload_device": device,
            "preparing_dtype": torch.float8_e4m3fn,
            "preparing_device": device,
            "computation_dtype": torch.bfloat16,
            "computation_device": device,
        }
    if offload:
        return {
            "offload_dtype": "disk",
            "offload_device": "disk",
            "onload_dtype": "disk",
            "onload_device": "disk",
            "preparing_dtype": torch.bfloat16,
            "preparing_device": device,
            "computation_dtype": torch.bfloat16,
            "computation_device": device,
            "clear_parameters": True,
        }
    return {}


def parse_model_configs(
    model_paths: Optional[str],
    model_id_with_origin_paths: Optional[str],
    fp8_models: Optional[str],
    offload_models: Optional[str],
    device: str,
) -> List[ModelConfig]:
    fp8_model_set = set(split_model_specs(fp8_models))
    offload_model_set = set(split_model_specs(offload_models))
    model_configs: List[ModelConfig] = []

    if model_paths is not None:
        paths = json.loads(model_paths)
        for path in paths:
            vram_config = parse_vram_config(
                fp8=path in fp8_model_set,
                offload=path in offload_model_set,
                device=device,
            )
            model_configs.append(ModelConfig(path=path, **vram_config))

    if model_id_with_origin_paths is not None:
        specs = split_model_specs(model_id_with_origin_paths)
        for spec in specs:
            parsed = parse_path_or_model_id(spec)
            vram_config = parse_vram_config(
                fp8=spec in fp8_model_set,
                offload=spec in offload_model_set,
                device=device,
            )
            model_configs.append(
                ModelConfig(
                    model_id=parsed.model_id,
                    origin_file_pattern=parsed.origin_file_pattern,
                    path=parsed.path,
                    **vram_config,
                )
            )

    return model_configs


def parse_qwen35_vram_config(
    qwen35_processor_path: Optional[str],
    qwen35_processor_config: Optional[ModelConfig],
    fp8_models: Optional[str],
    offload_models: Optional[str],
    device: str,
) -> Optional[Dict]:
    if qwen35_processor_path is None:
        return None
    fp8_model_set = set(split_model_specs(fp8_models))
    offload_model_set = set(split_model_specs(offload_models))
    qwen35_specs = {qwen35_processor_path}
    if os.path.exists(qwen35_processor_path):
        qwen35_specs.add(os.path.abspath(qwen35_processor_path))
    if qwen35_processor_config is not None:
        if qwen35_processor_config.path is not None:
            qwen35_specs.add(qwen35_processor_config.path)
        if qwen35_processor_config.model_id is not None:
            qwen35_specs.add(qwen35_processor_config.model_id)
    qwen35_fp8 = any(spec in fp8_model_set for spec in qwen35_specs)
    qwen35_offload = any(spec in offload_model_set for spec in qwen35_specs)
    if not qwen35_fp8 and not qwen35_offload:
        return None
    return parse_vram_config(fp8=qwen35_fp8, offload=qwen35_offload, device=device)


def parse_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def round_up_to_factor(value: int, factor: int) -> int:
    return (value + factor - 1) // factor * factor


def cuda_fp8_add_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    if not hasattr(torch, "float8_e4m3fn"):
        return False
    try:
        x = torch.zeros((1,), device="cuda", dtype=torch.float8_e4m3fn)
        y = torch.zeros((1,), device="cuda", dtype=torch.float8_e4m3fn)
        _ = x + y
        return True
    except Exception:
        return False


def collect_images(src_dir: Path, recursive: bool) -> List[Path]:
    iterator = src_dir.rglob("*") if recursive else src_dir.glob("*")
    image_paths = [path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
    image_paths.sort()
    return image_paths


def prefixed_path(path: Path, prefix: str) -> Path:
    return path.with_name(f"{prefix}{path.name}")


def build_org_filename_map(org_dir: Path) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = {}
    for path in org_dir.rglob("*"):
        if not (path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
            continue
        mapping.setdefault(path.name, []).append(path)
    return mapping


def resolve_checkpoint(path_or_dir: str, patterns: List[str], excludes: Optional[List[str]] = None) -> Path:
    path = Path(path_or_dir)
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Checkpoint path must be a file or directory: {path}")

    excludes = excludes or []
    candidates: List[Path] = []
    for pattern in patterns:
        candidates.extend([p for p in path.glob(pattern) if p.is_file()])
    candidates = [p for p in candidates if not any(p.name.endswith(s) for s in excludes)]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint matched in {path}. patterns={patterns}, excludes={excludes}"
        )
    candidates.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return candidates[-1]


def derive_aligner_checkpoint_from_lora(lora_path: Path) -> Optional[Path]:
    if lora_path.suffix != ".safetensors":
        return None
    if lora_path.name.endswith("-lora.safetensors"):
        candidate = lora_path.with_name(lora_path.name.replace("-lora.safetensors", "-aligner.safetensors"))
        return candidate if candidate.exists() else None
    candidate = lora_path.with_name(lora_path.stem + "-aligner.safetensors")
    return candidate if candidate.exists() else None


def extract_aligner_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    for prefix in ("pipe.qwen35_prompt_aligner.", "qwen35_prompt_aligner."):
        matched = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if matched:
            return matched
    return {}


def load_aligner_from_checkpoint(pipe: Flux2ImagePipeline, checkpoint_path: Path) -> bool:
    aligner = getattr(pipe, "qwen35_prompt_aligner", None)
    if aligner is None:
        raise RuntimeError("Current pipeline has no `qwen35_prompt_aligner`, cannot load aligner checkpoint.")

    state_dict = load_state_dict(str(checkpoint_path), device="cpu")
    aligner_state_dict = extract_aligner_state_dict(state_dict)
    if len(aligner_state_dict) == 0:
        return False

    load_result = aligner.load_state_dict(aligner_state_dict, strict=False)
    if len(load_result.missing_keys) > 0:
        print(f"Warning, missing aligner keys: {load_result.missing_keys}")
    if len(load_result.unexpected_keys) > 0:
        print(f"Warning, unexpected aligner keys: {load_result.unexpected_keys}")
    return True


def init_pipeline(args, device: str) -> Flux2ImagePipeline:
    torch_dtype = parse_torch_dtype(args.dtype)
    model_configs = parse_model_configs(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        device=device,
    )
    qwen35_processor_config = parse_path_or_model_id(args.qwen35_processor_path)
    qwen35_vram_config = parse_qwen35_vram_config(
        qwen35_processor_path=args.qwen35_processor_path,
        qwen35_processor_config=qwen35_processor_config,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        device=device,
    )
    print(f"[Init] qwen35_vram_config={qwen35_vram_config}")

    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=None,
        qwen35_processor_config=qwen35_processor_config,
        qwen35_vram_config=qwen35_vram_config,
        vram_limit=args.vram_limit,
    )

    if qwen35_processor_config is not None and getattr(pipe, "text_encoder_qwen35", None) is None:
        raise RuntimeError(
            "Failed to initialize Qwen3.5 image-text encoder. "
            "Please check --qwen35_processor_path."
        )

    # In pure inference, Qwen3.5 encoder may stay on CPU because it is loaded
    # outside model_configs (without accelerate). Keep its current dtype and
    # only align device when needed.
    text_encoder_qwen35 = getattr(pipe, "text_encoder_qwen35", None)
    if text_encoder_qwen35 is not None:
        if not (hasattr(text_encoder_qwen35, "vram_management_enabled") and text_encoder_qwen35.vram_management_enabled):
            text_encoder_qwen35.eval()

    qwen35_prompt_aligner = getattr(pipe, "qwen35_prompt_aligner", None)
    if qwen35_prompt_aligner is not None:
        qwen35_prompt_aligner.to(device=device, dtype=torch_dtype)
        qwen35_prompt_aligner.eval()

    return pipe


def ensure_output_dir(dst_dir: Path, overwrite: bool):
    if dst_dir.exists() and not dst_dir.is_dir():
        raise ValueError(f"Destination path exists and is not a directory: {dst_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        has_files = any(dst_dir.iterdir())
        if has_files:
            raise ValueError(
                f"Destination directory is not empty: {dst_dir}. "
                "Use --overwrite_dst to allow writing into an existing directory."
            )


def run_inference(args):
    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    org_dir = None if not args.org_dir else Path(args.org_dir)
    if not src_dir.is_dir():
        raise ValueError(f"Source directory not found: {src_dir}")
    if org_dir is not None and not org_dir.is_dir():
        raise ValueError(f"Original directory not found: {org_dir}")
    ensure_output_dir(dst_dir, overwrite=args.overwrite_dst)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rand_device = args.rand_device if args.rand_device is not None else ("cuda" if str(device).startswith("cuda") else "cpu")
    print(f"[Init] device={device}, rand_device={rand_device}")
    print(f"[Init] Flux2 pipeline source: {flux2_image_sr_module.__file__}")

    if args.fp8_models is not None and not args.allow_fp8_inference:
        print("[Init] --fp8_models is provided but FP8 inference is disabled by default. Falling back to non-FP8.")
        args.fp8_models = None

    if args.fp8_models is not None and str(device).startswith("cuda") and not cuda_fp8_add_supported():
        print(
            "[Init] CUDA float8 add is not supported in current env, "
            "but this pipeline uses BF16 compute with FP8 storage. Keep --fp8_models enabled."
        )

    fp8_models_without_vae = disable_vae_fp8_for_inference(args.fp8_models)
    if fp8_models_without_vae != args.fp8_models:
        print("[Init] Disable VAE FP8 for inference because current CUDA env cannot run VAE batchnorm stats in float8.")
        args.fp8_models = fp8_models_without_vae

    pipe = init_pipeline(args, device=device)
    text_encoder_qwen35 = getattr(pipe, "text_encoder_qwen35", None)
    if text_encoder_qwen35 is not None:
        ref_param = next(text_encoder_qwen35.parameters())
        print(f"[Init] text_encoder_qwen35 device={ref_param.device}, dtype={ref_param.dtype}")

    lora_path = resolve_checkpoint(
        args.lora_checkpoint,
        patterns=["*-lora.safetensors", "*.safetensors"],
        excludes=["-aligner.safetensors"],
    )
    print(f"[Init] Loading LoRA: {lora_path}")
    # In FP8 inference, hotloaded LoRA patching can hit unsupported CUDA add
    # on float8 tensors. Fuse once up front so the model still keeps FP8
    # storage/runtime management without per-forward FP8 additions.
    pipe.load_lora(pipe.dit, str(lora_path), hotload=False)

    aligner_loaded = False
    aligner_source = None
    if getattr(pipe, "qwen35_prompt_aligner", None) is not None:
        if args.aligner_checkpoint is not None:
            aligner_source = resolve_checkpoint(
                args.aligner_checkpoint,
                patterns=["*-aligner.safetensors", "*.safetensors"],
            )
        else:
            aligner_source = derive_aligner_checkpoint_from_lora(lora_path)

        if aligner_source is not None:
            print(f"[Init] Loading aligner: {aligner_source}")
            aligner_loaded = load_aligner_from_checkpoint(pipe, aligner_source)
            if not aligner_loaded:
                print(f"Warning, no aligner weights found in: {aligner_source}")

        if not aligner_loaded:
            print(f"[Init] Trying to load aligner from LoRA file: {lora_path}")
            aligner_loaded = load_aligner_from_checkpoint(pipe, lora_path)
            if not aligner_loaded:
                print("Warning, aligner weights were not found. Using random-initialized aligner.")

    image_paths = collect_images(src_dir, recursive=args.recursive)
    if len(image_paths) == 0:
        raise ValueError(f"No images found in source directory: {src_dir}")
    print(f"[Run] Total images: {len(image_paths)}")
    org_filename_map = build_org_filename_map(org_dir) if org_dir is not None else None

    success = 0
    failed = 0
    org_missing = 0
    org_ambiguous = 0
    for index, image_path in enumerate(tqdm(image_paths, desc="Running inference")):
        relative = image_path.relative_to(src_dir)
        output_path = dst_dir / relative
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if args.output_ext is not None:
            output_path = output_path.with_suffix("." + args.output_ext.lower().lstrip("."))

        try:
            src_copy_path = prefixed_path(dst_dir / relative, "src_")
            src_copy_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, src_copy_path)

            org_source_path = None
            if org_dir is not None:
                org_by_relative = org_dir / relative
                if org_by_relative.is_file():
                    org_source_path = org_by_relative
                else:
                    same_name_paths = org_filename_map.get(relative.name, []) if org_filename_map is not None else []
                    if len(same_name_paths) == 1:
                        org_source_path = same_name_paths[0]
                    elif len(same_name_paths) > 1:
                        org_ambiguous += 1
                        print(
                            f"[Org] Ambiguous filename for {relative.name}, "
                            f"matched {len(same_name_paths)} files in {org_dir}. Skip org copy."
                        )

                if org_source_path is not None:
                    org_copy_path = prefixed_path(dst_dir / relative, "org_")
                    org_copy_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(org_source_path, org_copy_path)
                else:
                    org_missing += 1
                    print(f"[Org] Not found for {relative}, skip org copy.")

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_height = args.height if args.height is not None else image.height
                image_width = args.width if args.width is not None else image.width
                image_height = round_up_to_factor(image_height, 16)
                image_width = round_up_to_factor(image_width, 16)
                seed = None if args.seed is None else args.seed + index * args.seed_stride

                input_image_for_pipe = None
                if args.use_input_image:
                    if image.size != (image_width, image_height):
                        input_image_for_pipe = image.resize((image_width, image_height), Image.BICUBIC)
                    else:
                        input_image_for_pipe = image

                result = pipe(
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    cfg_scale=args.cfg_scale,
                    embedded_guidance=args.embedded_guidance,
                    input_image=input_image_for_pipe,
                    edit_image=image,
                    edit_image_auto_resize=args.edit_image_auto_resize,
                    encode_image=image,
                    encode_image_auto_resize=args.encode_image_auto_resize,
                    height=image_height,
                    width=image_width,
                    seed=seed,
                    rand_device=rand_device,
                    num_inference_steps=args.num_inference_steps,
                )
                result.save(output_path)
                success += 1
        except (UnidentifiedImageError, OSError) as error:
            failed += 1
            print(f"[Skip] {image_path}: {error}")
        except Exception as error:
            failed += 1
            print(f"[Error] {image_path}: {error}")
            if os.environ.get("IMAGESR_DEBUG", "0") == "1":
                print(traceback.format_exc())

    print(f"[Done] success={success}, failed={failed}, org_missing={org_missing}, org_ambiguous={org_ambiguous}, dst={dst_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description="Batch inference for ImageSR model (LoRA + aligner).")
    parser.add_argument("--src_dir", type=str, required=True, help="Source directory containing only images.")
    parser.add_argument("--dst_dir", type=str, required=True, help="Destination directory to save generated images.")
    parser.add_argument(
        "--org_dir",
        type=str,
        default=None,
        help="Optional directory containing original images with same relative path/name for comparison copy.",
    )

    parser.add_argument("--model_paths", type=str, default=None, help="Model paths in JSON list format.")
    parser.add_argument(
        "--model_id_with_origin_paths",
        type=str,
        default=(
            "black-forest-labs/FLUX.2-klein-base-4B:transformer/*.safetensors,"
            "black-forest-labs/FLUX.2-klein-base-4B:vae/diffusion_pytorch_model.safetensors"
        ),
        help="Comma-separated model specs in format `model_id:origin_file_pattern`.",
    )
    parser.add_argument(
        "--qwen35_processor_path",
        type=str,
        default="./models/Jackrong/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled",
        help="Qwen3.5 processor model path or model id.",
    )
    parser.add_argument(
        "--fp8_models",
        type=str,
        default=None,
        help="Comma-separated specs to run in fp8 (same format as model spec).",
    )
    parser.add_argument(
        "--allow_fp8_inference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable FP8 inference path explicitly. Default is disabled for stability.",
    )
    parser.add_argument("--offload_models", type=str, default=None, help="Comma-separated specs to enable offload.")

    parser.add_argument(
        "--lora_checkpoint",
        type=str,
        default="./outputs/train",
        help="LoRA checkpoint file or directory. If directory, latest matching file is used.",
    )
    parser.add_argument(
        "--aligner_checkpoint",
        type=str,
        default=None,
        help="Aligner checkpoint file or directory. If empty, auto-derive from LoRA path.",
    )

    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Prompt used for all images.")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt.")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Denoising steps.")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="CFG scale.")
    parser.add_argument("--embedded_guidance", type=float, default=4.0, help="Embedded guidance value.")
    parser.add_argument("--height", type=int, default=None, help="Optional fixed output height.")
    parser.add_argument("--width", type=int, default=None, help="Optional fixed output width.")
    parser.add_argument("--seed", type=int, default=42, help="Base seed. Use --seed -1 to disable fixed seed.")
    parser.add_argument("--seed_stride", type=int, default=1, help="Per-image seed increment.")

    parser.add_argument("--device", type=str, default="auto", help="`auto`, `cuda`, `cpu`, or `cuda:0`.")
    parser.add_argument("--rand_device", type=str, default=None, help="Random generation device for noise.")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16", help="Torch dtype.")
    parser.add_argument("--vram_limit", type=float, default=None, help="Optional VRAM limit in GB.")

    parser.add_argument("--recursive", action="store_true", help="Recursively load images from src_dir.")
    parser.add_argument("--overwrite_dst", action="store_true", help="Allow writing into non-empty dst dir.")
    parser.add_argument("--output_ext", type=str, default=None, help="Force output extension, e.g. png/jpg.")
    parser.add_argument(
        "--use_input_image",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use source image as `input_image`.",
    )
    parser.add_argument(
        "--edit_image_auto_resize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable auto-resize for edit image.",
    )
    parser.add_argument(
        "--encode_image_auto_resize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable auto-resize for encode image.",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.seed is not None and args.seed < 0:
        args.seed = None
    if (args.height is None) != (args.width is None):
        raise ValueError("Please set both --height and --width together, or leave both empty.")
    run_inference(args)
