import torch
from diffusers import Flux2KleinPipeline
from torchao.quantization import quantize_
from torchao.quantization import Float8DynamicActivationFloat8WeightConfig
from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig


pipe = Flux2KleinPipeline.from_pretrained(
    "/root/imggen/flux9b/models/FLUX/FLUX.2-klein-base-4B", 
    torch_dtype=torch.bfloat16,
)
quantize_config = NVFP4DynamicActivationNVFP4WeightConfig()


def quantize_Flux2KleinPipeline(module):
    def quantize_Transformer(module):
        def quantize_Flux2TransformerBlock(module):
            def quantize_Flux2Attention(module):
                quantize_(module.to_q, quantize_config)
                quantize_(module.to_k, quantize_config)
                quantize_(module.to_v, quantize_config)
                # quantize_(module.to_out[0], quantize_config)
                quantize_(module.add_q_proj, quantize_config)
                quantize_(module.add_k_proj, quantize_config)
                quantize_(module.add_v_proj, quantize_config)
                # quantize_(module.to_add_out, quantize_config)
            quantize_Flux2Attention(module.attn)
            quantize_(module.ff, quantize_config)
            quantize_(module.ff_context, quantize_config)
        for m in module.transformer_blocks:
            quantize_Flux2TransformerBlock(m)
        # quantize_(module.double_stream_modulation_img, quantize_config)
        # quantize_(module.double_stream_modulation_txt, quantize_config)
        # quantize_(module.single_stream_modulation, quantize_config)
        # quantize_(module.time_guidance_embed, quantize_config)
        quantize_(module.single_transformer_blocks, quantize_config)
    quantize_Transformer(module.transformer)
    quantize_(module.text_encoder.model, quantize_config)


quantize_Flux2KleinPipeline(pipe)
pipe.to("cuda")
pipe = torch.compile(pipe, mode="max-autotune")


prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=50,
    generator=torch.Generator(device="cuda").manual_seed(0)
).images[0]
image.save("flux-klein.png")
