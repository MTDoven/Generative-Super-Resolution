import math
from dataclasses import dataclass, field

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


@dataclass
class AutoEncoderParams:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 32


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor):
        pad = (0, 1, 0, 1)
        x = nn.functional.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ):
        super().__init__()
        self.quant_conv = torch.nn.Conv2d(2 * z_channels, 2 * z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        h = self.quant_conv(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ):
        super().__init__()
        self.post_quant_conv = torch.nn.Conv2d(z_channels, z_channels, 1)
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        z = self.post_quant_conv(z)

        # get dtype for proper tracing
        upscale_dtype = next(self.up.parameters()).dtype

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # cast to proper dtype
        h = h.to(upscale_dtype)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.params = params
        self.encoder = Encoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )
        self.decoder = Decoder(
            resolution=params.resolution,
            in_channels=params.in_channels,
            ch=params.ch,
            out_ch=params.out_ch,
            ch_mult=params.ch_mult,
            num_res_blocks=params.num_res_blocks,
            z_channels=params.z_channels,
        )

        self.bn_eps = 1e-4
        self.bn_momentum = 0.1
        self.ps = [2, 2]
        self.bn = torch.nn.BatchNorm2d(
            math.prod(self.ps) * params.z_channels,
            eps=self.bn_eps,
            momentum=self.bn_momentum,
            affine=False,
            track_running_stats=True,
        )
        self.spatial_compression_ratio = 2 ** (len(params.ch_mult) - 1) * self.ps[0]
        self.tile_sample_min_size = self._align_tile_size(params.resolution, self.spatial_compression_ratio)
        self.tile_sample_overlap = self._align_tile_size(
            max(self.spatial_compression_ratio, min(self.tile_sample_min_size // 4, 64)),
            self.spatial_compression_ratio,
        )

    @staticmethod
    def _align_tile_size(size: int, scale: int) -> int:
        if size <= scale:
            return scale
        return max((size // scale) * scale, scale)

    @staticmethod
    def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
        if tile_size >= length:
            return [0]

        stride = max(tile_size - overlap, 1)
        starts = [0]
        while starts[-1] + tile_size < length:
            next_start = starts[-1] + stride
            if next_start + tile_size >= length:
                next_start = length - tile_size
            if next_start == starts[-1]:
                break
            starts.append(next_start)
        return starts

    @staticmethod
    def _tile_weight(
        height: int,
        width: int,
        overlap_h: int,
        overlap_w: int,
        top_edge: bool,
        bottom_edge: bool,
        left_edge: bool,
        right_edge: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        weight_y = torch.ones(height, device=device, dtype=dtype)
        weight_x = torch.ones(width, device=device, dtype=dtype)

        if overlap_h > 0:
            if not top_edge:
                ramp = torch.linspace(0, 1, overlap_h + 2, device=device, dtype=dtype)[1:-1]
                weight_y[:overlap_h] = ramp
            if not bottom_edge:
                ramp = torch.linspace(1, 0, overlap_h + 2, device=device, dtype=dtype)[1:-1]
                weight_y[-overlap_h:] = torch.minimum(weight_y[-overlap_h:], ramp)

        if overlap_w > 0:
            if not left_edge:
                ramp = torch.linspace(0, 1, overlap_w + 2, device=device, dtype=dtype)[1:-1]
                weight_x[:overlap_w] = ramp
            if not right_edge:
                ramp = torch.linspace(1, 0, overlap_w + 2, device=device, dtype=dtype)[1:-1]
                weight_x[-overlap_w:] = torch.minimum(weight_x[-overlap_w:], ramp)

        return weight_y[:, None] * weight_x[None, :]

    def normalize(self, z):
        self.bn.eval()
        return self.bn(z)

    def inv_normalize(self, z):
        self.bn.eval()
        s = torch.sqrt(self.bn.running_var.view(1, -1, 1, 1) + self.bn_eps)
        m = self.bn.running_mean.view(1, -1, 1, 1)
        return z * s + m

    def encode(self, x: Tensor) -> Tensor:
        moments = self.encoder(x)
        mean = torch.chunk(moments, 2, dim=1)[0]

        z = rearrange(
            mean,
            "... c (i pi) (j pj)  -> ... (c pi pj) i j",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        z = self.normalize(z)
        return z

    def tile_encode(
        self,
        x: Tensor,
        tile_sample_min_size: int | None = None,
        tile_sample_overlap: int | None = None,
    ) -> Tensor:
        tile_sample_min_size = self._align_tile_size(
            tile_sample_min_size or self.tile_sample_min_size,
            self.spatial_compression_ratio,
        )
        tile_sample_overlap = self._align_tile_size(
            tile_sample_overlap or self.tile_sample_overlap,
            self.spatial_compression_ratio,
        )

        _, _, height, width = x.shape
        if height <= tile_sample_min_size and width <= tile_sample_min_size:
            return self.encode(x)

        tile_sample_overlap = min(tile_sample_overlap, tile_sample_min_size - self.spatial_compression_ratio)
        latent_scale = self.spatial_compression_ratio
        latent_height = height // latent_scale
        latent_width = width // latent_scale
        tile_latent_size = tile_sample_min_size // latent_scale
        tile_latent_overlap = tile_sample_overlap // latent_scale

        output = None
        weights = None

        y_starts = self._tile_starts(height, min(tile_sample_min_size, height), tile_sample_overlap)
        x_starts = self._tile_starts(width, min(tile_sample_min_size, width), tile_sample_overlap)

        for y in y_starts:
            for x0 in x_starts:
                y_end = min(y + tile_sample_min_size, height)
                x_end = min(x0 + tile_sample_min_size, width)
                tile = x[:, :, y:y_end, x0:x_end]
                tile_latent = self.encode(tile)

                if output is None:
                    output = torch.zeros(
                        (tile_latent.shape[0], tile_latent.shape[1], latent_height, latent_width),
                        device=tile_latent.device,
                        dtype=tile_latent.dtype,
                    )
                    weights = torch.zeros_like(output)

                latent_y = y // latent_scale
                latent_x = x0 // latent_scale
                latent_y_end = latent_y + tile_latent.shape[-2]
                latent_x_end = latent_x + tile_latent.shape[-1]
                weight = self._tile_weight(
                    height=tile_latent.shape[-2],
                    width=tile_latent.shape[-1],
                    overlap_h=min(tile_latent_overlap, tile_latent.shape[-2] - 1),
                    overlap_w=min(tile_latent_overlap, tile_latent.shape[-1] - 1),
                    top_edge=latent_y == 0,
                    bottom_edge=latent_y_end == latent_height,
                    left_edge=latent_x == 0,
                    right_edge=latent_x_end == latent_width,
                    device=tile_latent.device,
                    dtype=tile_latent.dtype,
                )[None, None]

                output[:, :, latent_y:latent_y_end, latent_x:latent_x_end] = (
                    output[:, :, latent_y:latent_y_end, latent_x:latent_x_end] + tile_latent * weight
                )
                weights[:, :, latent_y:latent_y_end, latent_x:latent_x_end] = (
                    weights[:, :, latent_y:latent_y_end, latent_x:latent_x_end] + weight
                )

        return output / weights.clamp_min(torch.finfo(output.dtype).eps)

    def decode(self, z: Tensor) -> Tensor:
        z = self.inv_normalize(z)
        z = rearrange(
            z,
            "... (c pi pj) i j -> ... c (i pi) (j pj)",
            pi=self.ps[0],
            pj=self.ps[1],
        )
        dec = self.decoder(z)
        return dec

    def forward(self, x: Tensor, mode: str = "encode", tiled: bool = False) -> Tensor:
        """
        FSDP-safe entrypoint.

        Calling `encode()` / `decode()` directly can bypass FSDP pre-forward hooks when
        this module is wrapped, which may leave parameters in sharded/flattened form.
        Route all training/inference calls through `forward` instead.
        """
        if mode == "encode":
            return self.tile_encode(x) if tiled else self.encode(x)
        if mode == "decode":
            return self.tile_decode(x) if tiled else self.decode(x)
        raise ValueError(f"Unsupported mode '{mode}'. Expected 'encode' or 'decode'.")

    def tile_decode(
        self,
        z: Tensor,
        tile_sample_min_size: int | None = None,
        tile_sample_overlap: int | None = None,
    ) -> Tensor:
        tile_sample_min_size = self._align_tile_size(
            tile_sample_min_size or self.tile_sample_min_size,
            self.spatial_compression_ratio,
        )
        tile_sample_overlap = self._align_tile_size(
            tile_sample_overlap or self.tile_sample_overlap,
            self.spatial_compression_ratio,
        )

        latent_tile_size = tile_sample_min_size // self.spatial_compression_ratio
        _, _, latent_height, latent_width = z.shape
        if latent_height <= latent_tile_size and latent_width <= latent_tile_size:
            return self.decode(z)

        latent_tile_overlap = max(1, tile_sample_overlap // self.spatial_compression_ratio)
        latent_tile_overlap = min(latent_tile_overlap, latent_tile_size - 1)
        sample_height = latent_height * self.spatial_compression_ratio
        sample_width = latent_width * self.spatial_compression_ratio

        output = None
        weights = None

        y_starts = self._tile_starts(latent_height, min(latent_tile_size, latent_height), latent_tile_overlap)
        x_starts = self._tile_starts(latent_width, min(latent_tile_size, latent_width), latent_tile_overlap)

        for y in y_starts:
            for x0 in x_starts:
                y_end = min(y + latent_tile_size, latent_height)
                x_end = min(x0 + latent_tile_size, latent_width)
                tile = z[:, :, y:y_end, x0:x_end]
                
                # Checkpoint the decode step to save VRAM across tiles
                if getattr(z, 'requires_grad', False):
                    tile_image = checkpoint(self.decode, tile, use_reentrant=True)
                else:
                    tile_image = self.decode(tile)

                if output is None:
                    output = torch.zeros(
                        (tile_image.shape[0], tile_image.shape[1], sample_height, sample_width),
                        device=tile_image.device,
                        dtype=tile_image.dtype,
                    )
                    weights = torch.zeros_like(output)

                sample_y = y * self.spatial_compression_ratio
                sample_x = x0 * self.spatial_compression_ratio
                sample_y_end = sample_y + tile_image.shape[-2]
                sample_x_end = sample_x + tile_image.shape[-1]
                weight = self._tile_weight(
                    height=tile_image.shape[-2],
                    width=tile_image.shape[-1],
                    overlap_h=min(
                        latent_tile_overlap * self.spatial_compression_ratio,
                        tile_image.shape[-2] - 1,
                    ),
                    overlap_w=min(
                        latent_tile_overlap * self.spatial_compression_ratio,
                        tile_image.shape[-1] - 1,
                    ),
                    top_edge=sample_y == 0,
                    bottom_edge=sample_y_end == sample_height,
                    left_edge=sample_x == 0,
                    right_edge=sample_x_end == sample_width,
                    device=tile_image.device,
                    dtype=tile_image.dtype,
                )[None, None]

                output[:, :, sample_y:sample_y_end, sample_x:sample_x_end] = (
                    output[:, :, sample_y:sample_y_end, sample_x:sample_x_end] + tile_image * weight
                )
                weights[:, :, sample_y:sample_y_end, sample_x:sample_x_end] = (
                    weights[:, :, sample_y:sample_y_end, sample_x:sample_x_end] + weight
                )

        return output / weights.clamp_min(torch.finfo(output.dtype).eps)
