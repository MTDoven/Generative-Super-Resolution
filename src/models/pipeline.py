import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as T
from safetensors.torch import load_file
from einops import rearrange
from PIL import Image

from .autoencoder import AutoEncoder, AutoEncoderParams
from .transformer import Flux2, Klein4BParams
from .text_encoder import Qwen3VLEmbedder
from ..utils import checkpoint_transfer_diffusers2vae, checkpoint_transfer_diffusers2flux, get_dt_scale


def _get_x_ids(x: torch.Tensor) -> torch.Tensor:
    """Generate image token ids in [t, h, w, l] format."""
    b, _, h, w = x.shape
    h_coords = torch.arange(h, device=x.device, dtype=torch.long)
    w_coords = torch.arange(w, device=x.device, dtype=torch.long)
    hw = torch.cartesian_prod(h_coords, w_coords)  # (h*w, 2)
    zeros = torch.zeros((h * w, 1), device=x.device, dtype=torch.long)
    x_ids_single = torch.cat([zeros, hw, zeros], dim=1)  # (h*w, 4) => [t, h, w, l]
    x_ids = x_ids_single.unsqueeze(0).expand(b, -1, -1)
    return x_ids

def _get_ctx_ids(ctx: torch.Tensor) -> torch.Tensor:
    """Generate text token ids in [t, h, w, l] format."""
    b, l, _ = ctx.shape
    l_coords = torch.arange(l, device=ctx.device, dtype=torch.long)
    zeros = torch.zeros((l,), device=ctx.device, dtype=torch.long)
    ctx_ids_single = torch.stack([zeros, zeros, zeros, l_coords], dim=1)  # (l, 4)
    ctx_ids = ctx_ids_single.unsqueeze(0).expand(b, -1, -1)
    return ctx_ids

def _preprocess_images(images: list[Image.Image], dtype: torch.dtype = torch.float32, device: torch.device = torch.device('cpu')):
    to_tensor = T.ToTensor()
    processed = []
    for img in images:
        img = img.convert("RGB")
        tensor = to_tensor(img) * 2 - 1
        processed.append(tensor)
    batch = torch.stack(processed, dim=0).to(dtype=dtype, device=device)
    return batch

def _postprocess_images(images: torch.Tensor):
    images = images.detach().cpu()
    images = (images + 1) / 2
    images = images.clamp(0, 1).float()
    to_pil = T.ToPILImage()
    processed = []
    for i in range(images.shape[0]):
        img = to_pil(images[i])
        processed.append(img)
    return processed


class TextProjector(nn.Module):
    def __init__(self, in_features: int, out_features: int, feedforward_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        # Self-attention layer (Transformer layer style)
        self.attn = nn.MultiheadAttention(embed_dim=in_features, num_heads=8, batch_first=True)
        self.attn_norm = nn.LayerNorm(in_features)
        # FFN to transform feature dimensions
        self.in_proj = nn.Linear(in_features, feedforward_dim)
        self.out_proj = nn.Linear(feedforward_dim, out_features)
        # Residual projection matching
        self.res_proj = nn.Linear(in_features, out_features)
        self.activation = nn.SiLU()

    def forward(self, x):
        # 1. Residual path for projection
        x_res = self.res_proj(x)
        # 2. Attention Block
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out  # Residual around attention
        # 3. FFN Block for dimension projection
        x = self.attn_norm(x)
        x = self.in_proj(x)
        x = self.activation(x)
        x = self.out_proj(x)
        # Final residual connection
        return x + x_res
    

class ImageProjector(nn.Module):
    def __init__(self, in_features: int, out_features: int, feedforward_dim: int):
        super().__init__()
        mid_channels = feedforward_dim
        self.enc1 = nn.Conv2d(in_features, mid_channels, kernel_size=3, stride=1, padding=1)
        self.enc2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.enc3 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dec1 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.dec2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.dec3 = nn.Conv2d(mid_channels, out_features, kernel_size=3, stride=1, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.activation = nn.SiLU()

    def forward(self, x):
        x_in = x
        e1 = self.activation(self.enc1(x))
        p1 = self.pool(e1)
        e2 = self.activation(self.enc2(p1))
        p2 = self.pool(e2)
        e3 = self.activation(self.enc3(p2))
        d1 = self.up(e3) + e2
        d1 = self.activation(self.dec1(d1))
        d2 = self.up(d1) + e1
        d2 = self.activation(self.dec2(d2))
        d3 = self.activation(self.dec3(d2))
        return d3 + x_in


def _compute_image_loss(autoencoder, use_tiled_vae, latents, target_images):
    def _decode_and_loss(latents_slice, target_slice):
        gen = autoencoder(latents_slice, mode="decode", tiled=use_tiled_vae)
        return F.smooth_l1_loss(gen, target_slice)
    # Compute image loss in a memory-efficient way by processing one sample at a time
    image_loss = torch.tensor(0.0, device=latents.device, dtype=latents.dtype)
    b = latents.shape[0]
    for i in range(b):
        slice_loss = _decode_and_loss(
            latents[i:i+1],
            target_images[i:i+1]
        )  # Compute loss for the i-th sample
        image_loss = image_loss + slice_loss / b
    return image_loss

def _compute_alignment_loss(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    norm1 = F.normalize(emb1, p=2, dim=-1)
    norm2 = F.normalize(emb2, p=2, dim=-1)
    sim_matrix = torch.bmm(norm1, norm2.transpose(1, 2))
    max_sim_1 = sim_matrix.max(dim=2)[0]  # [B, L1]
    max_sim_2 = sim_matrix.max(dim=1)[0]  # [B, L2]
    return (1.0 - max_sim_1).mean() + (1.0 - max_sim_2).mean()


class Pipeline(nn.Module):
    def __init__(
        self, 
        autoencoder_ckpt: str,
        transformer_ckpt: str,
        text_encoder_ckpt: str,
        extra_hidden_dim: dict[str, int] = {},
        extra_loss_weight: dict[str, float] = {},
        velocity_scale: float = 1.0,
        use_tiled_vae: bool = False,
    ):
        """
        Modules: 
        autoencoder, transformer, text_encoder
        ctx_projector, in_image_projector, out_image_projector
        """
        super().__init__()
        # Initialize the autoencoder
        self.autoencoder = AutoEncoder(AutoEncoderParams())
        autoencoder_params = checkpoint_transfer_diffusers2vae(load_file(autoencoder_ckpt))
        self.autoencoder.load_state_dict(autoencoder_params, strict=True)
        # Initialize the transformer
        self.transformer = Flux2(Klein4BParams())
        transformer_params = checkpoint_transfer_diffusers2flux(load_file(transformer_ckpt))
        self.transformer.load_state_dict(transformer_params, strict=True)
        self.transformer.convert_layernorm_elementwise_affine_(verbose=False)
        # Initialize the text encoder
        self.text_encoder = Qwen3VLEmbedder(model_name_or_path=text_encoder_ckpt)
        # Projector to align text encoder output dimension with transformer input dimension
        self.extra_projector = nn.ModuleList([
            TextProjector(2048, 7680, feedforward_dim=extra_hidden_dim.get("ctx_projector", 2048)),
            ImageProjector(128, 128, feedforward_dim=extra_hidden_dim.get("in_image_projector", 512)),
            ImageProjector(128, 128, feedforward_dim=extra_hidden_dim.get("out_image_projector", 512)),
        ])
        # Velocity scale for controlling the strength of the velocity field
        self.extra_loss_weight = extra_loss_weight
        self.velocity_scale = velocity_scale
        self.use_tiled_vae = use_tiled_vae

    def _encode_images(self, images: list[Image.Image]):
        documents = [{"image": i} for i in images]
        out, last_hidden_state = self.text_encoder.process(
            documents, return_last_hidden_state=True)
        return out, last_hidden_state
    
    def _encode_texts(self, texts: list[str]):
        documents = [{"text": t} for t in texts]
        out, last_hidden_state = self.text_encoder.process(
            documents, return_last_hidden_state=True)
        return out, last_hidden_state

    def _compute_training_loss(
        self,
        latents: torch.Tensor,
        ctx: torch.Tensor,
        gt_texts: list[str] | None,
        gt_images: list[Image.Image] | None,
    ) -> dict[str, torch.Tensor]:
        # Encode the ground truth images and texts
        with torch.no_grad():
            _, gt_ctx = self._encode_texts(gt_texts)
            gt_ctx = self.extra_projector[0](gt_ctx)
            target_images = _preprocess_images(gt_images, dtype=latents.dtype, device=latents.device)
            target_latents = self.autoencoder(target_images, mode="encode", tiled=self.use_tiled_vae)
        
        # Compute losses
        text_loss = _compute_alignment_loss(
            ctx, 
            gt_ctx,
        )
        latent_loss = F.mse_loss(
            latents, 
            target_latents
        )
        image_loss = _compute_image_loss(
            self.autoencoder, 
            self.use_tiled_vae, 
            latents, 
            target_images
        )  # Compute image loss

        # Combine losses with respective weights
        total_loss = (
            text_loss * self.extra_loss_weight["text_encoder"]
            + latent_loss * self.extra_loss_weight["latent"]
            + image_loss * self.extra_loss_weight["image"]
        )  # Compute the total loss
        return {
            "total_loss": total_loss,
            "text_loss": text_loss,
            "latent_loss": latent_loss,
            "image_loss": image_loss
        }

    def forward(
        self, 
        images: list[Image.Image], 
        gt_images: list[Image.Image] = None, 
        gt_texts: list[str] = None,
        mode: str = "train",
    ) -> torch.Tensor | list[Image.Image]:
        if mode == "infer":
            return self.infer(images)

        # Encode the input images and texts
        _, ctx = self._encode_images(images)
        # Encode the input images into the latent space
        images = _preprocess_images(images, dtype=ctx.dtype, device=ctx.device)
        x = self.autoencoder(images, mode="encode", tiled=self.use_tiled_vae)
        b, c, h, w = x.shape
        # Compute position IDs
        x_ids = _get_x_ids(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        ctx_ids = _get_ctx_ids(ctx)
        ctx = self.extra_projector[0](ctx)
        # Integrate the context into the transformer
        velocity = self.transformer(
            x=x,
            x_ids=x_ids,
            timesteps=torch.full((b,), 0.0, device=x.device, dtype=x.dtype),
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=torch.full((b,), 4.0, device=x.device, dtype=x.dtype),
        )  # Compute the velocity
        dt_scale = get_dt_scale(h * 16, w * 16)
        x = x - dt_scale * self.velocity_scale * velocity
        # Decode the output back to the image space
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        # Compute training losses
        loss_dict = self._compute_training_loss(
            latents=x,
            ctx=ctx,
            gt_texts=gt_texts,
            gt_images=gt_images,
        )  # Compute the total training loss
        return loss_dict

    @torch.inference_mode()
    def infer(
        self, 
        images: list[Image.Image]
    ) -> list[Image.Image]:
        # Encode the input images and texts
        _, ctx = self._encode_images(images)
        # Encode the input images into the latent space
        images = _preprocess_images(images, dtype=ctx.dtype, device=ctx.device)
        x = self.autoencoder(images, mode="encode", tiled=self.use_tiled_vae)
        b, c, h, w = x.shape
        # Compute position IDs
        x_ids = _get_x_ids(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        ctx_ids = _get_ctx_ids(ctx)
        ctx = self.extra_projector[0](ctx)
        # Integrate the context into the transformer
        velocity = self.transformer(
            x=x,
            x_ids=x_ids,
            timesteps=torch.full((b,), 0.0, device=x.device, dtype=x.dtype),
            ctx=ctx,
            ctx_ids=ctx_ids,
            guidance=torch.full((b,), 4.0, device=x.device, dtype=x.dtype),
        )  # Compute the velocity
        dt_scale = get_dt_scale(h * 16, w * 16)
        x = x - dt_scale * self.velocity_scale * velocity
        # Decode the output back to the image space
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        gen_images = self.autoencoder(x, mode="decode", tiled=self.use_tiled_vae)
        # Post-process the generated images for inference
        gen_images = _postprocess_images(gen_images)
        return gen_images
