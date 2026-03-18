ROOT_PATH = "/root/super_resolution/Edit-SR"
from src.engine import EngineConfig, Engine
from src.data.dataset import ImageDataset


def main():

    config = EngineConfig(
        pipeline={
            "autoencoder_ckpt": f"{ROOT_PATH}/checkpoints/FLUX.2-Klein-Base-4B/vae/diffusion_pytorch_model.safetensors",
            "transformer_ckpt": f"{ROOT_PATH}/checkpoints/FLUX.2-Klein-Base-4B/transformer/diffusion_pytorch_model.safetensors",
            "text_encoder_ckpt": f"{ROOT_PATH}/checkpoints/Qwen3-VL-Embedding-2B",
            "extra_hidden_dim": {
                "ctx_projector": 2048,
                "in_image_projector": 512,
                "out_image_projector": 512,
            },
            "extra_loss_weight": {
                "text_encoder": 0.1,
                "latent": 0.3,
                "image": 0.6,
            },
            "velocity_scale": 3.0,
            "use_tiled_vae": True,
        },
        dataset={
            'class': ImageDataset,
            'train': {
                "metadata_file": f"{ROOT_PATH}/datasets/images/train_metadata.jsonl",
                "image_hw": (768, 768),
                "split": 0.2,
            },
            'test': {
                "metadata_file": f"{ROOT_PATH}/datasets/images/test_metadata.jsonl",
                "image_hw": (768, 768),
                "split": 0.2,
            },
        },
        trainer={
            "batch_size": 2,
            "num_workers": 4,
            "lr": 3e-4,
            "num_epochs": 10,
            "save_interval": 1,
            "image_save_path": f"{ROOT_PATH}/example_results",
            "checkpoint_save_path": f"{ROOT_PATH}/output_results",
            "device": "cuda",
            "gradient_accumulation_steps": 8,
            "mixed_precision": "bf16",
            "deepspeed_plugin": {
                "zero_stage": 3,
                "zero3_init_flag": True,
                "zero3_save_16bit_model": True,
                "offload_optimizer_device": "cpu",
                "offload_param_device": "none",
            }
        },
        trainable_modules=[
            "extra_projector",
        ],
        wandb={
            "project": "Edit-SR",
            "name": "super_resolution_training",
        }
    )

    engine = Engine(config)
    engine.run_training()




if __name__ == "__main__":
    main()
