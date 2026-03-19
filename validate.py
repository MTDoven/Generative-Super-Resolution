import re
import json
import torch
from PIL import Image
from pathlib import Path
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig


# --- Simple global configuration (edit as needed) ---
METADATA_PATH = Path("dataset/text_images/metadata_tiny.json")
OUT_DIR = Path("outputs/validate/finetune")
LORA_PATH = Path("outputs/train/finetune/epoch-4.safetensors")
MAX_ITEMS = 10  # set to an int to limit items, or None to run all
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42  # fixed seed (must be an int)


def load_metadata(path: Path):
    # assume the file is a JSON array of objects: [{"prompt": "...", ...}, ...]
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def init_pipeline(device: str, dtype=None):
    vram_config = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.float8_e4m3fn,
        "onload_device": "cpu",
        "preparing_dtype": torch.float8_e4m3fn,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    pipe = Flux2ImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="text_encoder/*.safetensors", **vram_config),
            ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="transformer/*.safetensors", **vram_config),
            ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
        ],
        tokenizer_config=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="tokenizer/"),
        vram_limit=torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 0.5,
    )
    return pipe


def run_generation(
    metadata_path: Path, 
    lora_path: Path, 
    output_path: Path, 
    max_items: int, 
):
    output_path.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(metadata_path)
    pipe = init_pipeline()
    assert lora_path is not None and lora_path.exists(), f"LoRA file not found: {lora_path}"
    #pipe.load_lora(pipe.dit, str(lora_path))

    # loop through metadata items and generate images
    total = len(meta) if max_items is None else min(len(meta), max_items)
    base_dir = metadata_path.parent
    for i in range(total):
        item = meta[i]
        # assume each item is a dict with `prompt` and `edit_image` or `image`
        prompt = "把图像变清晰"#item.get("prompt", "输出原图")

        # save input image (prefer edit_image[0] then image)
        input_rel = None
        if "edit_image" in item and isinstance(item["edit_image"], list) and item["edit_image"]:
            input_rel = item["edit_image"][0]
        elif "image" in item:
            input_rel = item["image"]

        # upsample input to 768x768 and use that size for generation
        target_size = (768, 768)
        width, height = target_size
        if input_rel is not None:
            input_path = base_dir / input_rel
            if input_path.exists():
                inp = Image.open(input_path).convert("RGB")
                up = inp.resize(target_size, Image.LANCZOS)
                in_snip = sanitize_filename(Path(input_rel).stem)[:40]
                in_name = f"{i:04d}_{in_snip}_input_768.jpg"
                up.save(output_path / in_name)

        # use fixed global SEED for generation with 768x768
        result = pipe(
            prompt=prompt, 
            seed=int(SEED), 
            height=height, 
            width=width
        )

        # assume result has a `save()` method per example usage
        snippet = sanitize_filename(prompt)[:40]
        seed_suffix = f"_s{SEED}"
        out_name = f"{i:04d}_{snippet}{seed_suffix}.jpg"
        output_path = output_path / out_name
        result.save(output_path)


if __name__ == "__main__":
    run_generation(METADATA_PATH, OUT_DIR, lora_path=LORA_PATH, max_items=MAX_ITEMS, device=DEVICE)

