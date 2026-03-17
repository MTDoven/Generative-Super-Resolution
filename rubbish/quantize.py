import torch
from torch import nn
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import NVFP4BlockScaling, Float8BlockScaling, Format


def replace_with_te_layers(module):
    for name, child in list(module.named_children()):
        if len(list(child.children())) > 0:
            replace_with_te_layers(child)

        # nn.Linear -> te.Linear
        if isinstance(child, nn.Linear):
            te_linear = te.Linear(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=(child.bias is not None),
                params_dtype=torch.bfloat16
            ).to("cpu")
            with torch.no_grad():
                te_linear.weight.copy_(child.weight)
                if child.bias is not None:
                    te_linear.bias.copy_(child.bias)
            setattr(module, name, te_linear)
            continue

        # nn.LayerNorm -> te.LayerNorm
        if isinstance(child, nn.LayerNorm):
            if isinstance(child.normalized_shape, tuple) and len(child.normalized_shape) != 1:
                continue
            te_norm = te.LayerNorm(
                hidden_size=(
                    child.normalized_shape[0]
                    if isinstance(child.normalized_shape, tuple)
                    else child.normalized_shape
                ),
                eps=child.eps,
                params_dtype=torch.bfloat16
            ).to("cpu")
            with torch.no_grad():
                if child.elementwise_affine:
                    te_norm.weight.copy_(child.weight)
                    if child.bias is not None:
                        te_norm.bias.copy_(child.bias)
            setattr(module, name, te_norm)
            continue

        # nn.RMSNorm -> te.RMSNorm
        if hasattr(te, "RMSNorm") and isinstance(child, nn.RMSNorm):
            if isinstance(child.normalized_shape, tuple) and len(child.normalized_shape) != 1:
                continue
            te_rmsnorm = te.RMSNorm(
                hidden_size=(
                    child.normalized_shape[0]
                    if isinstance(child.normalized_shape, tuple)
                    else child.normalized_shape
                ),
                eps=child.eps,
                params_dtype=torch.bfloat16
            ).to("cpu")
            with torch.no_grad():
                if child.elementwise_affine:
                    te_rmsnorm.weight.copy_(child.weight)
            setattr(module, name, te_rmsnorm)

    # After replacing all layers, return the modified module
    return module


def quantize_modules(modules_with_qname: list[tuple], training: bool = True):
    nvfp4_recipe = NVFP4BlockScaling(
        fp4_format=Format.E2M1,
        disable_rht=False,
        disable_stochastic_rounding=False if training else True,
        disable_2d_quantization=False,
        interval=1,
        history_len=512 if training else 1,
        scaling_factor_whitelist=["weight", "activation", "gradient"]
    )
    fp8_recipe = Float8BlockScaling(
        fp8_format=Format.E4M3,
        disable_rht=False,
        interval=1,
        history_len=512 if training else 1,
        disable_2d_quantization=False,
        scaling_factor_whitelist=["weight", "activation", "gradient"],
        use_stochastic_rounding=True if training else False,
    )
    # Apply quantization to the specified modules
    modules_with_recipe = []
    for module, recipe_name in modules_with_qname:
        replace_with_te_layers(module)
        if recipe_name == "fp16" or recipe_name == "bf16":
            this_recipe = {"enabled": False,}
        elif recipe_name == "nvfp4":
            this_recipe = {"enabled": True, "recipe": nvfp4_recipe}
        elif recipe_name == "fp8":
            this_recipe = {"enabled": True, "recipe": fp8_recipe}
        else:  # Unsupported recipe name
            raise ValueError(f"Unsupported recipe name: {recipe_name}")
        modules_with_recipe.append((module, this_recipe))
    return modules_with_recipe
