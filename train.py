import torch, os, argparse, accelerate
from diffsynth.core import UnifiedDataset, load_state_dict
from diffsynth.pipelines.flux2_image_sr import Flux2ImagePipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Flux2ImageTrainingModule(DiffusionTrainingModule):
    def merge_trainable_models(self, trainable_models, pipe):
        models = [] if trainable_models is None or trainable_models == "" else [name for name in trainable_models.split(",") if name != ""]
        if getattr(pipe, "qwen35_prompt_aligner", None) is not None and "qwen35_prompt_aligner" not in models:
            models.append("qwen35_prompt_aligner")
        return ",".join(models) if len(models) > 0 else None

    def load_extra_trainable_modules(self, checkpoint_path):
        aligner = getattr(self.pipe, "qwen35_prompt_aligner", None)
        if checkpoint_path is None or aligner is None:
            return

        state_dict = load_state_dict(checkpoint_path, device="cpu")
        aligner_state_dict = {}
        for prefix in ("pipe.qwen35_prompt_aligner.", "qwen35_prompt_aligner."):
            matched = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if len(matched) > 0:
                aligner_state_dict = matched
                break

        if len(aligner_state_dict) == 0:
            return

        load_result = aligner.load_state_dict(aligner_state_dict, strict=False)
        if len(load_result.missing_keys) > 0:
            print(f"Warning, missing keys when loading qwen35_prompt_aligner: {load_result.missing_keys}")
        if len(load_result.unexpected_keys) > 0:
            print(f"Warning, unexpected keys when loading qwen35_prompt_aligner: {load_result.unexpected_keys}")

    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        # tokenizer_path=None,  # legacy path, disabled
        qwen35_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
    ):
        super().__init__()
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        # tokenizer_config = self.parse_path_or_model_id(tokenizer_path, default_value=ModelConfig(model_id="black-forest-labs/FLUX.2-klein-base-4B", origin_file_pattern="tokenizer/"))
        tokenizer_config = None
        if qwen35_processor_path is None:
            qwen35_processor_config = None
        elif os.path.exists(qwen35_processor_path):
            qwen35_processor_config = ModelConfig(path=qwen35_processor_path)
        elif ":" in qwen35_processor_path:
            qwen35_processor_config = self.parse_path_or_model_id(qwen35_processor_path)
        else:
            qwen35_processor_config = ModelConfig(model_id=qwen35_processor_path, origin_file_pattern=None)
        self.pipe = Flux2ImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=device,
            model_configs=model_configs,
            tokenizer_config=tokenizer_config,
            qwen35_processor_config=qwen35_processor_config,
        )
        if qwen35_processor_path is not None and getattr(self.pipe, "text_encoder_qwen35", None) is None:
            raise ValueError(
                "Failed to load Qwen3.5 image-text encoder. "
                "Please ensure `--qwen35_processor_path` points to a valid model repo or local path."
            )
        trainable_models = self.merge_trainable_models(trainable_models, self.pipe)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)

        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        self.load_extra_trainable_modules(lora_checkpoint)
        
        # Other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        
    def get_pipeline_inputs(self, data):
        edit_image = data["edit_image"] if "edit_image" in data else None
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {"negative_prompt": ""}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_image": data["image"],
            "edit_image": edit_image,
            "edit_image_auto_resize": True,
            "encode_image": edit_image,
            "encode_image_auto_resize": True,
            "height": data["image"].size[1],
            "width": data["image"].size[0],
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "embedded_guidance": 1.0,
            "cfg_scale": 1,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        # Keep Qwen3.5 image-text input fully aligned with edit-image input.
        inputs_shared["encode_image"] = inputs_shared.get("edit_image")
        inputs_shared["encode_image_auto_resize"] = inputs_shared.get("edit_image_auto_resize", True)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def flux2_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_image_size_config(parser)
    # parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")  # legacy path, disabled
    parser.add_argument("--qwen35_processor_path", type=str, default=None, help="Path to Qwen3.5 image-text processor (AutoProcessor).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    return parser


if __name__ == "__main__":
    parser = flux2_parser()
    args = parser.parse_args()
    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_image_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
        )
    )
    model = Flux2ImageTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        # tokenizer_path=args.tokenizer_path,  # legacy path, disabled
        qwen35_processor_path=args.qwen35_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task,
        "sft:train": launch_training_task,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
