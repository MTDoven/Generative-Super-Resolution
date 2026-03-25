import os, torch
from accelerate import Accelerator


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        self.num_steps += 1
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = unwrapped_model.export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}-lora.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)
            aligner = self.get_qwen35_prompt_aligner(unwrapped_model)
            if aligner is not None:
                self.save_aligner_state(accelerator, aligner, f"epoch-{epoch_id}-aligner.safetensors")


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = unwrapped_model.export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)
            aligner = self.get_qwen35_prompt_aligner(unwrapped_model)
            if aligner is not None:
                aligner_file_name = file_name.replace(".safetensors", "-aligner.safetensors")
                self.save_aligner_state(accelerator, aligner, aligner_file_name)


    def get_qwen35_prompt_aligner(self, model: torch.nn.Module):
        pipe = getattr(model, "pipe", None)
        aligner = getattr(pipe, "qwen35_prompt_aligner", None)
        if aligner is None or aligner.__class__.__name__ != "Qwen35PromptAligner":
            return None
        return aligner


    def save_aligner_state(self, accelerator: Accelerator, aligner: torch.nn.Module, file_name):
        aligner_state_dict = {
            f"qwen35_prompt_aligner.{key}": value.detach().cpu()
            for key, value in aligner.state_dict().items()
        }
        path = os.path.join(self.output_path, file_name)
        accelerator.save(aligner_state_dict, path, safe_serialization=True)
