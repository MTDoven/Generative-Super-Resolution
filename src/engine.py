import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import math
import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from accelerate import Accelerator, DeepSpeedPlugin
from .models.pipeline import Pipeline
from dataclasses import dataclass
import gc
import time
from functools import wraps
import wandb


def _empty_cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        gc.collect()
        time.sleep(0.1)
        torch.cuda.empty_cache()
        result = func(*args, **kwargs)
        gc.collect()
        time.sleep(0.1)
        torch.cuda.empty_cache()
        return result
    return wrapper


@dataclass
class EngineConfig:
    pipeline: dict[str, any]
    dataset: dict[str, any]  
    trainer: dict[str, any]
    trainable_modules: list[str]
    wandb: dict[str, any] = None


class Engine(nn.Module):
    def __init__(self, config: EngineConfig):
        super().__init__()
        self.config = config
        self.pipeline = Pipeline(**config.pipeline)
        self.train_dataset = config.dataset['class'](**config.dataset['train'])
        self.test_dataset = config.dataset['class'](**config.dataset['test'])
        # Get the actual module references for the trainable modules
        self.trainable_modules = [
            getattr(self.pipeline, module_name)
            for module_name in config.trainable_modules
        ]
        # Keep a uniform parameter dtype before FSDP wrapping.
        self.pipeline.to(dtype=torch.bfloat16)
        self.freeze_()
        self.train()
        # Initialize the accelerator
        ds_config = config.trainer["deepspeed_plugin"]
        deepspeed_plugin = DeepSpeedPlugin(
            zero_stage=ds_config["zero_stage"],
            zero3_init_flag=ds_config["zero3_init_flag"],
            zero3_save_16bit_model=ds_config["zero3_save_16bit_model"],
            offload_optimizer_device=ds_config["offload_optimizer_device"],
            offload_param_device=ds_config["offload_param_device"],
        )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.trainer["gradient_accumulation_steps"],
            mixed_precision=config.trainer["mixed_precision"],
            deepspeed_plugin=deepspeed_plugin,
        )
        # Build dataloader and optimizer after initializing the accelerator
        self.train_dataloader = self._build_train_dataloader()
        self.test_dataloader = self._build_test_dataloader()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.pipeline, self.optimizer, self.train_dataloader, self.test_dataloader, self.scheduler = self.accelerator.prepare(
            self.pipeline,
            self.optimizer,
            self.train_dataloader,
            self.test_dataloader,
            self.scheduler
        )
        # Initialize wandb if config provided
        if self.config.wandb:
            if self.accelerator.is_local_main_process:
                wandb.init(**self.config.wandb)
        # Print the initialized engine configuration for verification
        print("Engine initialized with configuration:", config)

    def _build_train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.trainer["batch_size"],
            num_workers=self.config.trainer["num_workers"],
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_dataset.collate_fn,
        )

    def _build_test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.config.trainer["batch_size"],
            num_workers=self.config.trainer["num_workers"],
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            collate_fn=self.test_dataset.collate_fn,
        )

    def _build_optimizer(self) -> Optimizer:
        return AdamW(
            filter(lambda p: p.requires_grad, self.pipeline.parameters()),
            lr=self.config.trainer["lr"],
        )

    def _build_scheduler(self):
        steps_per_epoch = math.ceil(len(self.train_dataloader) / self.config.trainer["gradient_accumulation_steps"])
        total_steps = steps_per_epoch * self.config.trainer["num_epochs"]
        return CosineAnnealingLR(self.optimizer, T_max=total_steps)

    def freeze_(self):
        trainable_module_ids = {id(module) for module in self.trainable_modules}
        # Freeze everything first
        for module in self.pipeline.modules():
            for param in module.parameters():
                param.requires_grad = False
        # Unfreeze selected modules
        for module in self.pipeline.modules():
            if id(module) in trainable_module_ids:
                for param in module.parameters():
                    param.requires_grad = True

    def _log_training_info(self, epoch: int, step: int, loss_dict: dict[str, torch.Tensor]):
        if self.accelerator.sync_gradients:
            reduced_losses = {}
            for key, loss in loss_dict.items():
                reduced_losses[key] = self.accelerator.reduce(loss.detach(), reduction="mean").item()
            if self.accelerator.is_local_main_process:
                current_lr = self.optimizer.param_groups[0]['lr']
                log = f"Epoch [{epoch + 1}/{self.config.trainer['num_epochs']}], " \
                      f"Step [{step + 1}/{len(self.train_dataloader)}], " \
                      f"Total Loss: {reduced_losses['total_loss']:.4f}, " \
                      f"Text Loss: {reduced_losses['text_loss']:.4f}, " \
                      f"Latent Loss: {reduced_losses['latent_loss']:.4f}, " \
                      f"Image Loss: {reduced_losses['image_loss']:.4f}, " \
                      f"LR: {current_lr:.6f}"
                print(log)
                # Log to wandb if initialized
                if self.config.wandb:
                    wandb_log = {
                        "epoch": epoch + 1,
                        "step": step + 1,
                        "total_loss": reduced_losses['total_loss'],
                        "text_loss": reduced_losses['text_loss'],
                        "latent_loss": reduced_losses['latent_loss'],
                        "image_loss": reduced_losses['image_loss'],
                        "learning_rate": current_lr
                    }
                    wandb.log(wandb_log)

    def _save_checkpoint(self, ckpt_save_dir: str):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_local_main_process:
            os.makedirs(ckpt_save_dir, exist_ok=True)
            print(f"Saving state to {ckpt_save_dir}/checkpoint ...")
        self.accelerator.save_state(f"{ckpt_save_dir}/checkpoint")
        self.accelerator.wait_for_everyone()

    @_empty_cache
    def _generate_examples(self, epoch_dir_name: str, num_examples: int = None):
        torch.cuda.empty_cache()
        # All ranks must participate in inference to avoid DeepSpeed ZeRO hanging
        # Construct full image save path inside the function
        img_save_dir = f"{self.config.trainer['image_save_path']}/{epoch_dir_name}"
        if self.accelerator.is_local_main_process:
            os.makedirs(img_save_dir, exist_ok=True)
            print("Generating examples...")
        # Switch pipeline to eval mode for generation
        self.pipeline.eval()
        with torch.inference_mode():
            batch = next(iter(self.test_dataloader))
            input_images = batch["lr_image"]
            gt_images = batch["hr_image"]
            # Use DeepSpeed interface via forward pass instead of unwrap
            out_images = self.pipeline(images=input_images, mode="infer")
            # Only save on main process
            if self.accelerator.is_local_main_process:
                for idx, (inp, gt) in enumerate(zip(input_images, gt_images)):
                    if num_examples is not None and idx >= num_examples:
                        break
                    inp.save(f"{img_save_dir}/example_{idx}_input.png")
                    gt.save(f"{img_save_dir}/example_{idx}_gt.png")
                    if out_images:
                        out_images[idx].save(f"{img_save_dir}/example_{idx}_pred.png")
                print(f"Examples saved to {img_save_dir}/")
        self.pipeline.train()

    def run_training(self):
        # Training loop only
        for epoch in range(self.config.trainer["num_epochs"]):

            # Save and generate examples at the specified interval
            if epoch % self.config.trainer["save_interval"] == 0:
                self._save_checkpoint(self.config.trainer['checkpoint_save_path'])
                self._generate_examples(f"epoch_{epoch + 1}")

            # Training step
            for i, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.pipeline):
                    with self.accelerator.autocast():
                        loss_dict = self.pipeline(
                            images=batch["lr_image"],
                            gt_images=batch["hr_image"],
                            gt_texts=batch["text"],
                        )
                        loss = loss_dict["total_loss"]
                    # Backpropagate the loss and update parameters
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                # Log training information
                self._log_training_info(epoch, i, loss_dict)

        # Final save and example generation after all epochs
        self._save_checkpoint(self.config.trainer['checkpoint_save_path'])
        self._generate_examples("epoch_final")
