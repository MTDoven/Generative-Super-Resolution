PYTORCH_ALLOC_CONF=expandable_segments:True \
accelerate launch train.py \
  --dataset_base_path ./dataset/text_images \
  --dataset_metadata_path ./dataset/text_images/metadata_edit_500.csv \
  --model_id_with_origin_paths "black-forest-labs/FLUX.2-klein-base-4B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-base-4B:vae/diffusion_pytorch_model.safetensors" \
  --qwen35_processor_path "./models/Jackrong/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled" \
  --fp8_models "black-forest-labs/FLUX.2-klein-base-4B:transformer/*.safetensors,black-forest-labs/FLUX.2-klein-base-4B:vae/diffusion_pytorch_model.safetensors" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out" \
  --lora_base_model "dit" \
  --data_file_keys "image,edit_image" \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --use_gradient_checkpointing \
  \
  --dataset_repeat 1 \
  --dataset_num_workers 8 \
  --output_path "./outputs/train" \
  --lora_rank 128 \
  --learning_rate 1e-4 \
  --gradient_accumulation_steps 4 \
  --num_epochs 5 \
