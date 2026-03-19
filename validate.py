import re
import pandas as pd
import torch
from PIL import Image
from pathlib import Path
from diffsynth.pipelines.flux2_image import Flux2ImagePipeline, ModelConfig


METADATA_PATH = Path("dataset/text_images/metadata_edit_500.csv")
OUTPUT_PATH = Path("outputs/validate")
LORA_PATH = Path("outputs/train/epoch-4.safetensors")
MAX_ITEMS = 50
SEED = 42


def load_metadata(path: Path):
    return pd.read_csv(path).to_dict('records')


def init_pipeline():
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


def generate(
    metadata_path: Path, 
    lora_path: Path, 
    output_path: Path, 
    max_items: int, 
):
    output_path.mkdir(parents=True, exist_ok=True)
    meta = load_metadata(metadata_path)
    pipe = init_pipeline()
    pipe.load_lora(pipe.dit, str(lora_path))

    total = min(len(meta), max_items)
    base_dir = metadata_path.parent
    for i in range(total):
        item = meta[i]
        prompt = item["prompt"]
        image = Image.open(base_dir / item["image"]).convert("RGB")
        edit_image = Image.open(base_dir / item["edit_image"]).convert("RGB").resize(image.size)
        print(f"Processing item {i}: "
              f"edit_image: {edit_image.height}x{edit_image.width}; "
              f"image: {image.height}x{image.width}")
        # Generate image with the pipeline, using the edit_image and prompt from metadata
        gen_image = pipe(
            edit_image=edit_image,
            prompt=prompt, 
            seed=int(SEED), 
            height=edit_image.height, 
            width=edit_image.width,
            cfg_scale=4.0,
            num_inference_steps=50,
        )
        gen_name = Path(item["image"]).stem + "_gen" + Path(item["image"]).suffix
        org_name = Path(item["image"]).stem + "_org" + Path(item["image"]).suffix
        ipt_name = Path(item["image"]).stem + "_ipt" + Path(item["image"]).suffix
        gen_image.save(output_path / gen_name)
        image.save(output_path / org_name)
        edit_image.save(output_path / ipt_name)


if __name__ == "__main__":
    generate(METADATA_PATH, LORA_PATH, OUTPUT_PATH, MAX_ITEMS)

