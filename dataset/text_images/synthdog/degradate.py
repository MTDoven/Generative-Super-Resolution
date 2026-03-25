"""
Image Degradation Pipeline
Following Real-ESRGAN and SeeSR style degradation methods.
High-order degradation with resize to 256-512 short edge.

Usage:
    python degradation.py --src /path/to/high_res_images --dst /path/to/save_degraded --cores 8
"""

import os
import argparse
import cv2
import numpy as np
import random
import math
from multiprocessing import Pool
from tqdm import tqdm
from scipy import special
from scipy.stats import multivariate_normal
from scipy.ndimage import convolve


# ==================== Kernel Generation (Real-ESRGAN style) ====================

def sigma_matrix2(sig_x, sig_y, theta):
    """Calculate the rotated sigma matrix (covariance matrix)."""
    d_matrix = np.array([[sig_x ** 2, 0], [0, sig_y ** 2]])
    u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))


def mesh_grid(kernel_size):
    """Generate mesh grid for kernel."""
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.stack([xx, yy], axis=-1)
    return xy


def pdf2(sigma_matrix, grid):
    """Calculate probability density function of 2D Gaussian."""
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, axis=-1))
    return kernel


def bivariate_gaussian(kernel_size, sig_x, sig_y, theta, isotropic=True):
    """Generate a bivariate isotropic or anisotropic Gaussian kernel."""
    grid = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x ** 2, 0], [0, sig_x ** 2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    kernel = pdf2(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)
    return kernel


def bivariate_generalized_gaussian(kernel_size, sig_x, sig_y, theta, beta, isotropic=True):
    """Generate a bivariate generalized Gaussian kernel (Real-ESRGAN)."""
    grid = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x ** 2, 0], [0, sig_x ** 2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.power(np.sum(np.dot(grid, inverse_sigma) * grid, axis=-1), beta))
    kernel = kernel / np.sum(kernel)
    return kernel


def bivariate_plateau(kernel_size, sig_x, sig_y, theta, beta, isotropic=True):
    """Generate a plateau-shaped kernel (Real-ESRGAN)."""
    grid = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x ** 2, 0], [0, sig_x ** 2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = 1 / (np.power(np.sum(np.dot(grid, inverse_sigma) * grid, axis=-1), beta) + 1)
    kernel = kernel / np.sum(kernel)
    return kernel


def random_bivariate_gaussian(kernel_size, sigma_x_range, sigma_y_range, rotation_range, isotropic_prob=0.5):
    """Randomly generate bivariate gaussian kernel."""
    if random.random() < isotropic_prob:
        sigma_x = random.uniform(*sigma_x_range)
        sigma_y = sigma_x
        rotation = 0
    else:
        sigma_x = random.uniform(*sigma_x_range)
        sigma_y = random.uniform(*sigma_y_range)
        rotation = random.uniform(*rotation_range)
    
    kernel = bivariate_gaussian(kernel_size, sigma_x, sigma_y, rotation, isotropic=(sigma_x == sigma_y))
    return kernel


def random_bivariate_generalized_gaussian(kernel_size, sigma_x_range, sigma_y_range, rotation_range, beta_range, isotropic_prob=0.5):
    """Randomly generate bivariate generalized Gaussian kernel."""
    if random.random() < isotropic_prob:
        sigma_x = random.uniform(*sigma_x_range)
        sigma_y = sigma_x
        rotation = 0
    else:
        sigma_x = random.uniform(*sigma_x_range)
        sigma_y = random.uniform(*sigma_y_range)
        rotation = random.uniform(*rotation_range)
    
    beta = random.uniform(*beta_range)
    kernel = bivariate_generalized_gaussian(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=(sigma_x == sigma_y))
    return kernel


def random_bivariate_plateau(kernel_size, sigma_x_range, sigma_y_range, rotation_range, beta_range, isotropic_prob=0.5):
    """Randomly generate plateau-shaped kernel."""
    if random.random() < isotropic_prob:
        sigma_x = random.uniform(*sigma_x_range)
        sigma_y = sigma_x
        rotation = 0
    else:
        sigma_x = random.uniform(*sigma_x_range)
        sigma_y = random.uniform(*sigma_y_range)
        rotation = random.uniform(*rotation_range)
    
    beta = random.uniform(*beta_range)
    if beta < 1e-3:
        beta = 1e-3
    kernel = bivariate_plateau(kernel_size, sigma_x, sigma_y, rotation, beta, isotropic=(sigma_x == sigma_y))
    return kernel


def circular_lowpass_kernel(cutoff, kernel_size, pad_to=0):
    """Generate 2D sinc filter (circular lowpass filter) - causes ringing artifacts."""
    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            r = np.sqrt(x**2 + y**2)
            if r == 0:
                kernel[i, j] = cutoff ** 2
            else:
                kernel[i, j] = cutoff * special.j1(cutoff * r * np.pi) / (2 * r)
    
    kernel = kernel / np.sum(kernel)
    
    if pad_to > kernel_size:
        pad_size = (pad_to - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))
    
    return kernel


def random_mixed_kernels(kernel_list, kernel_prob, kernel_size, sigma_x_range, sigma_y_range, 
                          rotation_range, beta_gaussian_range, beta_plateau_range, isotropic_prob=0.5):
    """Randomly select and generate a kernel from the kernel list."""
    kernel_type = random.choices(kernel_list, kernel_prob)[0]
    
    if kernel_type == 'iso':
        kernel = random_bivariate_gaussian(kernel_size, sigma_x_range, sigma_y_range, rotation_range, isotropic_prob=1.0)
    elif kernel_type == 'aniso':
        kernel = random_bivariate_gaussian(kernel_size, sigma_x_range, sigma_y_range, rotation_range, isotropic_prob=0.0)
    elif kernel_type == 'generalized_iso':
        kernel = random_bivariate_generalized_gaussian(kernel_size, sigma_x_range, sigma_y_range, rotation_range, beta_gaussian_range, isotropic_prob=1.0)
    elif kernel_type == 'generalized_aniso':
        kernel = random_bivariate_generalized_gaussian(kernel_size, sigma_x_range, sigma_y_range, rotation_range, beta_gaussian_range, isotropic_prob=0.0)
    elif kernel_type == 'plateau_iso':
        kernel = random_bivariate_plateau(kernel_size, sigma_x_range, sigma_y_range, rotation_range, beta_plateau_range, isotropic_prob=1.0)
    elif kernel_type == 'plateau_aniso':
        kernel = random_bivariate_plateau(kernel_size, sigma_x_range, sigma_y_range, rotation_range, beta_plateau_range, isotropic_prob=0.0)
    elif kernel_type == 'sinc':
        cutoff = random.uniform(0.1, 0.5) * (kernel_size // 2)
        kernel = circular_lowpass_kernel(cutoff, kernel_size)
    else:
        kernel = random_bivariate_gaussian(kernel_size, sigma_x_range, sigma_y_range, rotation_range, isotropic_prob)
    
    return kernel


# ==================== Degradation Functions ====================

def filter2D(img, kernel):
    """Apply 2D filter to image."""
    kernel = kernel.astype(np.float32)
    return cv2.filter2D(img, -1, kernel)


def random_add_gaussian_noise(img, sigma_range=(1, 30), gray_prob=0.4):
    """Add Gaussian noise to image."""
    sigma = random.uniform(*sigma_range)
    
    if random.random() < gray_prob:
        # Gray noise (same noise for all channels)
        noise = np.random.randn(img.shape[0], img.shape[1]) * sigma
        noise = np.stack([noise] * img.shape[2], axis=-1)
    else:
        # Color noise (different noise for each channel)
        noise = np.random.randn(*img.shape) * sigma
    
    img_float = img.astype(np.float32) + noise
    return np.clip(img_float, 0, 255).astype(np.uint8)


def random_add_poisson_noise(img, scale_range=(0.05, 3.0), gray_prob=0.4):
    """Add Poisson noise to image."""
    scale = random.uniform(*scale_range)
    
    if random.random() < gray_prob:
        # Gray noise
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        noise = np.random.poisson(gray * 255 * scale).astype(np.float32) / (scale * 255) - gray
        noise = np.stack([noise] * 3, axis=-1) * 255
    else:
        # Color noise
        img_float = img.astype(np.float32) / 255.0
        noise = np.random.poisson(img_float * 255 * scale).astype(np.float32) / (scale * 255) - img_float
        noise = noise * 255
    
    img_noisy = img.astype(np.float32) + noise
    return np.clip(img_noisy, 0, 255).astype(np.uint8)


def random_add_speckle_noise(img, sigma_range=(0.01, 0.15)):
    """Add speckle (multiplicative) noise."""
    sigma = random.uniform(*sigma_range)
    img_float = img.astype(np.float32) / 255.0
    noise = np.random.randn(*img.shape) * sigma
    noisy = img_float + img_float * noise
    return np.clip(noisy * 255, 0, 255).astype(np.uint8)


def random_jpeg_compression(img, quality_range=(30, 95)):
    """Apply JPEG compression artifacts."""
    quality = random.randint(*quality_range)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg


def random_webp_compression(img, quality_range=(30, 95)):
    """Apply WebP compression artifacts."""
    quality = random.randint(*quality_range)
    encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
    _, encimg = cv2.imencode('.webp', img, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg


def random_resize(img, scale_range, resize_prob=[0.2, 0.7, 0.1]):
    """
    Random resize with different interpolation methods.
    resize_prob: [up_prob, down_prob, keep_prob]
    """
    h, w = img.shape[:2]
    
    resize_type = random.choices(['up', 'down', 'keep'], resize_prob)[0]
    
    if resize_type == 'up':
        scale = random.uniform(1, scale_range[1])
    elif resize_type == 'down':
        scale = random.uniform(scale_range[0], 1)
    else:
        return img
    
    # Random interpolation method
    interpolations = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
    interp_probs = [0.05, 0.3, 0.3, 0.3, 0.05]
    interpolation = random.choices(interpolations, interp_probs)[0]
    
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    
    return img


def final_resize(img, target_short_range=(256, 512)):
    """Resize image so short edge is within target range."""
    h, w = img.shape[:2]
    short_edge = min(h, w)
    target_short = random.randint(*target_short_range)

    # Use rounded size to avoid float precision + floor causing 1024 -> 1023.
    scale = target_short / float(short_edge)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    # Guarantee exact short edge size requested by args (e.g. 1024).
    if new_h <= new_w:
        new_h = target_short
    if new_w <= new_h:
        new_w = target_short
    
    # Use random interpolation for final resize
    interpolations = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
    interpolation = random.choice(interpolations)
    
    img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    return img


def usm_sharp(img, weight=0.5, radius=50, threshold=10):
    """Unsharp masking sharpening."""
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img.astype(np.float32) - blur.astype(np.float32)
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype(np.float32)
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)
    sharpened = img.astype(np.float32) + weight * residual
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    return sharpened


# ==================== Degradation Pipeline (Real-ESRGAN style) ====================

def degradation_pipeline_v2(img, target_short_range=(256, 512)):
    """
    Real-ESRGAN style high-order degradation pipeline.
    Two-stage degradation process with final resize to target size.
    """
    
    # Configuration for first degradation
    kernel_list1 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    kernel_prob1 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    blur_sigma1 = [0.2, 3]
    blur_sigma2_1 = [0.2, 1.5]
    betag_range1 = [0.5, 4]
    betap_range1 = [1, 2]
    
    # Configuration for second degradation
    kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
    kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    blur_sigma2 = [0.2, 1.5]
    blur_sigma2_2 = [0.2, 1.2]
    betag_range2 = [0.5, 4]
    betap_range2 = [1, 2]
    
    sinc_prob = 0.1  # Probability of applying sinc filter
    
    # =============== First Degradation Stage ===============
    
    # 1. Blur
    kernel_size = random.choice([7, 9, 11, 13, 15, 17, 19, 21])
    if np.random.uniform() < sinc_prob:
        # Sinc filter (ringing artifacts)
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c / np.pi, kernel_size)
    else:
        kernel = random_mixed_kernels(
            kernel_list1, kernel_prob1, kernel_size,
            blur_sigma1, blur_sigma2_1, [-math.pi, math.pi],
            betag_range1, betap_range1
        )
    img = filter2D(img, kernel)
    
    # 2. Resize (down)
    scale = random.uniform(0.15, 1.5)
    interpolations = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
    interp_probs = [0.05, 0.3, 0.3, 0.3, 0.05]
    interpolation = random.choices(interpolations, interp_probs)[0]
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=interpolation)
    
    # 3. Noise
    if random.random() < 0.5:
        if random.random() < 0.5:
            img = random_add_gaussian_noise(img, sigma_range=(1, 30), gray_prob=0.4)
        else:
            img = random_add_poisson_noise(img, scale_range=(0.05, 2.5), gray_prob=0.4)
    
    # 4. JPEG compression
    if random.random() < 0.5:
        img = random_jpeg_compression(img, quality_range=(30, 95))
    
    # =============== Second Degradation Stage ===============
    
    # 1. Blur
    kernel_size = random.choice([7, 9, 11, 13, 15, 17, 19, 21])
    if np.random.uniform() < sinc_prob:
        if kernel_size < 13:
            omega_c = np.random.uniform(np.pi / 3, np.pi)
        else:
            omega_c = np.random.uniform(np.pi / 5, np.pi)
        kernel = circular_lowpass_kernel(omega_c / np.pi, kernel_size)
    else:
        kernel = random_mixed_kernels(
            kernel_list2, kernel_prob2, kernel_size,
            blur_sigma2, blur_sigma2_2, [-math.pi, math.pi],
            betag_range2, betap_range2
        )
    img = filter2D(img, kernel)
    
    # 2. Resize (random up/down/keep)
    scale = random.uniform(0.3, 1.2)
    interpolation = random.choices(interpolations, interp_probs)[0]
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=interpolation)
    
    # 3. Noise
    if random.random() < 0.5:
        if random.random() < 0.5:
            img = random_add_gaussian_noise(img, sigma_range=(1, 25), gray_prob=0.4)
        else:
            img = random_add_poisson_noise(img, scale_range=(0.05, 2.5), gray_prob=0.4)
    
    # 4. JPEG compression (or last sinc filter + JPEG)
    if random.random() < 0.5:
        # Sinc filter before final JPEG
        if random.random() < sinc_prob:
            kernel_size = random.choice([7, 9, 11, 13, 15, 17, 19, 21])
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            kernel = circular_lowpass_kernel(omega_c / np.pi, kernel_size)
            img = filter2D(img, kernel)
    
    # Final JPEG/WebP compression
    if random.random() < 0.7:
        if random.random() < 0.7:
            img = random_jpeg_compression(img, quality_range=(30, 95))
        else:
            img = random_webp_compression(img, quality_range=(30, 95))
    
    # =============== Final Resize to Target Size ===============
    img = final_resize(img, target_short_range)
    
    return img


def degradation_pipeline(img, target_short_range=(256, 512)):
    """Main degradation pipeline wrapper."""
    return degradation_pipeline_v2(img, target_short_range)


# ==================== Processing Functions ====================

def process_image(args):
    """Process a single image."""
    src_path, dst_path, target_short_range = args
    
    try:
        img = cv2.imread(src_path)
        if img is None:
            print(f"Failed to read {src_path}")
            return False
        
        # Apply degradation with resize to target size
        degraded_img = degradation_pipeline(img, target_short_range)
        
        # Ensure destination directory exists
        dst_dir = os.path.dirname(dst_path)
        if dst_dir and not os.path.exists(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
        
        # Save
        cv2.imwrite(dst_path, degraded_img)
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Apply real-world degradation to images (Real-ESRGAN style)."
    )
    parser.add_argument("--src", type=str, required=True, help="Source folder path")
    parser.add_argument("--dst", type=str, required=True, help="Destination folder path")
    parser.add_argument("--cores", type=int, default=4, help="Number of CPU cores to use")
    parser.add_argument("--short_min", type=int, default=192, help="Minimum short edge size")
    parser.add_argument("--short_max", type=int, default=320, help="Maximum short edge size")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.src):
        print(f"Error: Source folder '{args.src}' does not exist.")
        return
    
    if not os.path.exists(args.dst):
        os.makedirs(args.dst)
    
    target_short_range = (args.short_min, args.short_max)
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    tasks = []
    for root, dirs, files in os.walk(args.src):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                src_path = os.path.join(root, file)
                
                # Maintain relative folder structure
                rel_path = os.path.relpath(src_path, args.src)
                dst_path = os.path.join(args.dst, rel_path)
                
                tasks.append((src_path, dst_path, target_short_range))
    
    if not tasks:
        print("No images found in source folder.")
        return
    
    print(f"Found {len(tasks)} images. Processing with {args.cores} cores...")
    print(f"Output short edge range: {target_short_range[0]}-{target_short_range[1]} pixels")
    
    with Pool(args.cores) as p:
        results = list(tqdm(p.imap(process_image, tasks), total=len(tasks)))
    
    success_count = sum(results)
    print(f"Done. Successfully processed {success_count}/{len(tasks)} images.")


if __name__ == "__main__":
    main()
