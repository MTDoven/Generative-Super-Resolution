import re
import math
import torch
from collections import defaultdict


def checkpoint_transfer_diffusers2flux(diction):
    MAPPING_diffusers2flux = {
        "context_embedder.weight": "txt_in.weight",
        "double_stream_modulation_img.linear.weight": "double_stream_modulation_img.lin.weight",
        "double_stream_modulation_txt.linear.weight": "double_stream_modulation_txt.lin.weight",
        "norm_out.linear.weight": "final_layer.adaLN_modulation.1.weight",
        "proj_out.weight": "final_layer.linear.weight",
        "single_stream_modulation.linear.weight": "single_stream_modulation.lin.weight",
        "time_guidance_embed.timestep_embedder.linear_1.weight": "time_in.in_layer.weight",
        "time_guidance_embed.timestep_embedder.linear_2.weight": "time_in.out_layer.weight",
        "x_embedder.weight": "img_in.weight",
        "single_transformer_blocks.": {
            ".attn.": ".",
            "single_transformer_blocks": "single_blocks",
            "norm_k.weight": "norm.key_norm.scale",
            "norm_q.weight": "norm.query_norm.scale",
            "to_out.weight": "linear2.weight",
            "to_qkv_mlp_proj.weight": "linear1.weight",
        },
        "transformer_blocks.": {
            "transformer_blocks": "double_blocks",
            "attn.to_*.weight": "img_attn.qkv.weight___*",
            "attn.add_*_proj.weight": "txt_attn.qkv.weight___*",
            "attn.to_add_out.weight": "txt_attn.proj.weight",
            "attn.to_out.0.weight": "img_attn.proj.weight",
            "attn.norm_k.weight": "img_attn.norm.key_norm.scale",
            "attn.norm_q.weight": "img_attn.norm.query_norm.scale",
            "attn.norm_added_k.weight": "txt_attn.norm.key_norm.scale",
            "attn.norm_added_q.weight": "txt_attn.norm.query_norm.scale",
            "ff.linear_in.weight": "img_mlp.0.weight",
            "ff.linear_out.weight": "img_mlp.2.weight",
            "ff_context.linear_in.weight": "txt_mlp.0.weight",
            "ff_context.linear_out.weight": "txt_mlp.2.weight",
        },
    }

    def apply_replacements(k, replacements):
        # First, apply replacements without '*'
        for org, rplc in replacements.items():
            if '*' not in org:
                k = k.replace(org, rplc)
        # Then, apply replacements with '*'
        for org, rplc in replacements.items():
            if '*' in org:
                org_pattern = re.escape(org).replace(r'\*', r'(.+)')
                def repl_func(match):
                    groups = match.groups()
                    new_rplc = rplc
                    for g in groups:
                        new_rplc = new_rplc.replace('*', g, 1)
                    return new_rplc
                k = re.sub(org_pattern, repl_func, k)
        return k

    # Create a new dictionary with the updated keys based on the mapping
    new_diction = {}
    for k, v in diction.items():
        if k in MAPPING_diffusers2flux:
            k = MAPPING_diffusers2flux[k]
            new_diction[k] = v
            continue
        for i in MAPPING_diffusers2flux:
            if k.startswith(i):
                k = apply_replacements(k, MAPPING_diffusers2flux[i])
                new_diction[k] = v
                break
        else:  # if no break occurred
            raise KeyError(f"Key {k} not found in mapping")
        
    # Now we need to concatenate the qkv weights for the attention layers
    qkv_diction = defaultdict(list)
    to_del_list = []
    for k, v in new_diction.items():
        if "___" in k:
            to_del_list.append(k)
            k = k.split("___")[0]
            qkv_diction[k].append(v)
    for k in to_del_list:
        del new_diction[k]
    for k, v in qkv_diction.items():
        assert len(v) == 3, f"Expected 3 tensors for {k}, but got {len(v)}."
        qkv_diction[k] = torch.cat(v, dim=0)

    # Update the original dictionary with the concatenated tensors
    new_diction.update(qkv_diction)
    return new_diction


def checkpoint_transfer_diffusers2vae(diction):
    MAPPING_diffusers2vae = {
        "down_blocks": "down",
        # The up_blocks are in reverse order in the original code...
        "up_blocks.3": "up.0",
        "up_blocks.2": "up.1",
        "up_blocks.1": "up.2",
        "up_blocks.0": "up.3",
        "downsamplers.0": "downsample",
        "upsamplers.0": "upsample",
        "resnets": "block",
        "mid_block": "mid",
        "attentions.0": "attn_1",
        "to_k": "k",
        "to_q": "q",
        "to_v": "v",
        "to_out.0": "proj_out",
        "mid.block.0": "mid.block_1",
        "mid.block.1": "mid.block_2",
        "group_norm": "norm",
        "conv_shortcut": "nin_shortcut",
        "conv_norm_out": "norm_out",
        # This is a special case, the encoder and decoder share the same "quant_conv"
        "quant_conv": "encoder.quant_conv",
        "post_quant_conv": "decoder.post_quant_conv",
        "post_encoder.quant_conv": "decoder.post_quant_conv",
    }
    SPECIAL_expansion = [
        "encoder.mid.attn_1.q.weight", 
        "encoder.mid.attn_1.k.weight", 
        "encoder.mid.attn_1.v.weight",
        "encoder.mid.attn_1.proj_out.weight",
        "decoder.mid.attn_1.q.weight",
        "decoder.mid.attn_1.k.weight",
        "decoder.mid.attn_1.v.weight",
        "decoder.mid.attn_1.proj_out.weight",
    ]
    new_diction = {}
    for k, v in diction.items():
        for i, j in MAPPING_diffusers2vae.items():
            k = k.replace(i, j)
        new_diction[k] = v
    for k, v in new_diction.items():
        if k in SPECIAL_expansion:
            v = v[..., None, None]
            new_diction[k] = v
    return new_diction


def get_dt_scale(h: int, w: int, num_steps: int = 50) -> float:
    """Compute FLUX dynamic-shifted last-step timestep for given image size."""
    image_seq_len = (h // 16) * (w // 16)
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666
    # Use different linear functions for image sequence lengths
    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
    else:
        m_200 = a2 * image_seq_len + b2
        m_10 = a1 * image_seq_len + b1
        a = (m_200 - m_10) / 190.0
        b = m_200 - 200.0 * a
        mu = a * num_steps + b
    # Compute the last step timestep using the logistic function
    t = 1.0 / num_steps
    if t == 0.0:
        return 0.0
    if t == 1.0:
        return 1.0
    return math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))
