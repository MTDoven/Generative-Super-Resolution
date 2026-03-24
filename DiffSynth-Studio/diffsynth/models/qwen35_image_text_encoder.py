from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig, Qwen3_5VisionConfig
import torch


class Qwen35ImageTextEncoder(torch.nn.Module):
	def __init__(self, model_size="4B"):
		super().__init__()

		def build_layer_types(num_hidden_layers):
			return ["full_attention" if (index + 1) % 4 == 0 else "linear_attention" for index in range(num_hidden_layers)]

		def build_text_config(
			hidden_size,
			intermediate_size,
			num_hidden_layers,
			num_attention_heads,
			num_key_value_heads,
			linear_num_value_heads,
		):
			return Qwen3_5TextConfig(
				vocab_size=248320,
				hidden_size=hidden_size,
				intermediate_size=intermediate_size,
				num_hidden_layers=num_hidden_layers,
				num_attention_heads=num_attention_heads,
				num_key_value_heads=num_key_value_heads,
				max_position_embeddings=262144,
				tie_word_embeddings=True,
				attention_bias=False,
				attention_dropout=0.0,
				head_dim=256,
				linear_conv_kernel_dim=4,
				linear_key_head_dim=128,
				linear_value_head_dim=128,
				linear_num_key_heads=16,
				linear_num_value_heads=linear_num_value_heads,
				rope_parameters={
					"mrope_interleaved": True,
					"mrope_section": [11, 11, 10],
					"rope_type": "default",
					"rope_theta": 10000000,
					"partial_rotary_factor": 0.25,
				},
				layer_types=build_layer_types(num_hidden_layers),
				attn_output_gate=True,
				dtype="bfloat16",
				mamba_ssm_dtype="float32",
				full_attention_interval=4,
				mtp_num_hidden_layers=1,
				mtp_use_dedicated_embeddings=False,
			)

		def build_vision_config(hidden_size, depth, intermediate_size, num_heads, out_hidden_size):
			return Qwen3_5VisionConfig(
				hidden_size=hidden_size,
				depth=depth,
				intermediate_size=intermediate_size,
				num_heads=num_heads,
				out_hidden_size=out_hidden_size,
				in_channels=3,
				num_position_embeddings=2304,
				patch_size=16,
				spatial_merge_size=2,
				temporal_patch_size=2,
				hidden_act="gelu_pytorch_tanh",
				initializer_range=0.02,
			)

		def build_multimodal_config(text_config, vision_config):
			config = Qwen3_5Config(
				image_token_id=248056,
				video_token_id=248057,
				vision_start_token_id=248053,
				vision_end_token_id=248054,
				tie_word_embeddings=True,
			)
			config.text_config = text_config
			config.vision_config = vision_config
			return config

		config_dict = {
			"0.8B": build_multimodal_config(
				build_text_config(1024, 3584, 24, 8, 2, 16),
				build_vision_config(768, 12, 3072, 12, 1024),
			),
			"2B": build_multimodal_config(
				build_text_config(2048, 6144, 24, 8, 2, 16),
				build_vision_config(1024, 24, 4096, 16, 2048),
			),
			"4B": build_multimodal_config(
				build_text_config(2560, 9216, 32, 16, 4, 32),
				build_vision_config(1024, 24, 4096, 16, 2560),
			),
		}

		if model_size not in config_dict:
			raise ValueError(f"Unsupported model_size: {model_size}. Expected one of: {', '.join(config_dict)}")

		self.model = Qwen3_5ForConditionalGeneration(config_dict[model_size])

	def forward(self, *args, **kwargs):
		return self.model(*args, **kwargs)
