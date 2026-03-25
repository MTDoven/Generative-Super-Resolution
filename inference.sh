PYTORCH_ALLOC_CONF=expandable_segments:True \
python inference.py \
  --src_dir "./dataset/text_images/lr_image" \
  --org_dir "./dataset/text_images/hr_image" \
  --dst_dir "./outputs/inference/full" \
  --overwrite_dst \
  --lora_checkpoint "./outputs/train/epoch-1-lora.safetensors" \
  --aligner_checkpoint "./outputs/train/epoch-1-aligner.safetensors" \
  --model_id_with_origin_paths "black-forest-labs/FLUX.2-klein-base-4B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-base-4B:vae/diffusion_pytorch_model.safetensors" \
  --qwen35_processor_path "./models/Qwen/Qwen3.5-4B" \
  --fp8_models "black-forest-labs/FLUX.2-klein-base-4B:transformer/*.safetensors,./models/Qwen/Qwen3.5-4B" \
  --allow_fp8_inference \
  --device "cuda" \
  --rand_device "cuda" \
  --cfg_scale 1.0 \
  --num_inference_steps 50 \
  --embedded_guidance 4.0 \
