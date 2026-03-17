import random
from io import BytesIO
from typing import Callable, Dict, List, Tuple
import numpy as np
from PIL import Image, ImageFilter
DegradeFn = Callable[[Image.Image], Image.Image]


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _degrade_gaussian_blur(image: Image.Image) -> Image.Image:
    radius = random.uniform(0.3, 2.5)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def _degrade_box_blur(image: Image.Image) -> Image.Image:
    radius = random.uniform(0.5, 2.0)
    return image.filter(ImageFilter.BoxBlur(radius=radius))


def _degrade_downsample_then_upsample(image: Image.Image) -> Image.Image:
    w, h = image.size
    scale = random.uniform(0.3, 0.85)
    small_w = max(1, int(round(w * scale)))
    small_h = max(1, int(round(h * scale)))
    # Randomly choose different resampling filters.
    down_filter = random.choice([
        Image.Resampling.BOX,
        Image.Resampling.BILINEAR,
        Image.Resampling.BICUBIC,
        Image.Resampling.LANCZOS,
    ])
    up_filter = random.choice([
        Image.Resampling.NEAREST,
        Image.Resampling.BILINEAR,
        Image.Resampling.BICUBIC,
    ])
    return image.resize((small_w, small_h), down_filter).resize((w, h), up_filter)


def _degrade_gaussian_noise(image: Image.Image) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    sigma = random.uniform(3.0, 20.0)
    noise = np.random.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _degrade_poisson_noise(image: Image.Image) -> Image.Image:
    arr = np.asarray(image).astype(np.float32) / 255.0
    peak = random.uniform(12.0, 60.0)
    noisy = np.random.poisson(arr * peak) / peak
    noisy = np.clip(noisy * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy, mode="RGB")


def _degrade_salt_pepper_noise(image: Image.Image) -> Image.Image:
    arr = np.asarray(image).copy()
    h, w, _ = arr.shape
    ratio = random.uniform(0.002, 0.02)
    num_pixels = int(h * w * ratio)
    ys = np.random.randint(0, h, size=num_pixels)
    xs = np.random.randint(0, w, size=num_pixels)
    salt_mask = np.random.rand(num_pixels) > 0.5
    arr[ys[salt_mask], xs[salt_mask], :] = 255
    arr[ys[~salt_mask], xs[~salt_mask], :] = 0
    return Image.fromarray(arr, mode="RGB")


def _degrade_jpeg_compression(image: Image.Image) -> Image.Image:
    quality = random.randint(20, 70)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _degrade_sharpen_overshoot(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.UnsharpMask(radius=random.uniform(1.0, 2.2), percent=random.randint(180, 320), threshold=0))


def _degrade_color_jitter(image: Image.Image) -> Image.Image:
    arr = np.asarray(image).astype(np.float32)
    gain = np.array([
        random.uniform(0.8, 1.2),
        random.uniform(0.8, 1.2),
        random.uniform(0.8, 1.2),
    ], dtype=np.float32)
    bias = random.uniform(-18.0, 18.0)
    arr = arr * gain.reshape(1, 1, 3) + bias
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


DEFAULT_DEGRADATIONS: Dict[str, DegradeFn] = {
    "gaussian_blur": _degrade_gaussian_blur,
    "box_blur": _degrade_box_blur,
    "down_up_sample": _degrade_downsample_then_upsample,
    "gaussian_noise": _degrade_gaussian_noise,
    "poisson_noise": _degrade_poisson_noise,
    "salt_pepper_noise": _degrade_salt_pepper_noise,
    "jpeg_compression": _degrade_jpeg_compression,
    "sharpen_overshoot": _degrade_sharpen_overshoot,
    "color_jitter": _degrade_color_jitter,
}

DEFAULT_DEGRADATION_WEIGHTS: Dict[str, float] = {
    "gaussian_blur": 1.4,
    "box_blur": 1.0,
    "down_up_sample": 1.8,
    "gaussian_noise": 1.3,
    "poisson_noise": 0.8,
    "salt_pepper_noise": 0.6,
    "jpeg_compression": 1.6,
    "sharpen_overshoot": 0.3,
    "color_jitter": 0.5,
}


def random_destroy_image(image: Image.Image, min_ops: int = 1, max_ops: int = 4) -> Image.Image:
    image = _ensure_rgb(image)
    all_ops = list(DEFAULT_DEGRADATIONS.keys())
    real_max_ops = min(max(max_ops, 1), len(all_ops))
    real_min_ops = min(max(min_ops, 1), real_max_ops)
    num_ops = random.randint(real_min_ops, real_max_ops)
    candidates: List[Tuple[str, float]] = [
        (name, max(DEFAULT_DEGRADATION_WEIGHTS.get(name, 1.0), 1e-6))
        for name in all_ops
    ]
    picked: List[str] = []
    for _ in range(num_ops):
        names = [name for name, _ in candidates]
        weights = [weight for _, weight in candidates]
        idx = random.choices(range(len(names)), weights=weights, k=1)[0]
        picked.append(names[idx])
        candidates.pop(idx)
    out = image
    for op_name in picked:
        out = DEFAULT_DEGRADATIONS[op_name](out)
    return out
