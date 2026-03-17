from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor, nn
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR


class Qwen3Embedder(nn.Module):
    def __init__(
        self,
        model_spec: str,
        device: str | torch.device = "cuda",
        max_length: int = MAX_LENGTH,
        min_pixels: int = MIN_PIXELS,
        max_pixels: int = MAX_PIXELS,
        normalize: bool = True,
    ):
        super().__init__()

        self.device = torch.device(device)
        self.max_length = max_length
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.normalize = normalize
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.model = Qwen3VLModel.from_pretrained(
            model_spec,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.processor = Qwen3VLProcessor.from_pretrained(
            model_spec,
            padding_side="right",
        )
        self.model.eval()

    @staticmethod
    def _pool_last(hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        flipped_mask = attention_mask.flip(dims=[1])
        last_positions = attention_mask.shape[1] - flipped_mask.argmax(dim=1) - 1
        batch_indices = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[batch_indices, last_positions]

    @staticmethod
    def _ensure_list(value, name: str):
        if isinstance(value, (str, Path, Image.Image)):
            return [value]
        if isinstance(value, Sequence):
            return list(value)
        raise TypeError(f"{name} must be a string, image, or a sequence of them.")

    @staticmethod
    def _load_image(image: str | Path | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        image_path = Path(image)
        with Image.open(image_path) as opened_image:
            return opened_image.convert("RGB")

    def _build_messages(self, texts: list[str]) -> list[list[dict]]:
        conversations = []
        for text in texts:
            conversations.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": text},
                        ],
                    }
                ]
            )
        return conversations

    @torch.no_grad()
    def forward(
        self,
        text: str | Sequence[str],
        image: str | Path | Image.Image | Sequence[str | Path | Image.Image],
        normalize: bool | None = None,
    ) -> Tensor:
        texts = self._ensure_list(text, "text")
        images = self._ensure_list(image, "image")

        if len(texts) != len(images):
            raise ValueError("text and image must have the same batch size.")

        conversations = self._build_messages(texts)
        prompt_text = self.processor.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_images = [self._load_image(item) for item in images]

        model_inputs = self.processor(
            text=prompt_text,
            images=prompt_images,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        model_inputs = {key: value.to(self.device) for key, value in model_inputs.items()}

        outputs = self.model(**model_inputs)
        embeddings = self._pool_last(outputs.last_hidden_state, model_inputs["attention_mask"])

        should_normalize = self.normalize if normalize is None else normalize
        if should_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings
