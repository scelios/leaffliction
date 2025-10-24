#!/usr/bin/env python3

import os
from pathlib import Path
import sys
import argparse
import skimage as ski
import numpy as np

import Utils as u


def _ensure_same_shape(orig, processed: np.ndarray):
    """Resize and clip the processed image to match the original shape and uint8 type."""
    if processed.shape != orig.shape:
        ret = ski.transform.resize(processed, orig.shape, anti_aliasing=True, preserve_range=False)
        return ski.util.img_as_ubyte(np.clip(ret, 0, 1)) # type: ignore
    return ski.util.img_as_ubyte(np.clip(processed, 0, 1))


def rotate(image, angle):
    """Rotate image by given angle, keeping same dimensions."""
    rotated = ski.transform.rotate(image, angle, resize=True, mode='constant', cval=1)
    return _ensure_same_shape(image, rotated)


def rescale(image, scale_factor):
    """Rescale image but keep output size same as input (crop/fill as needed)."""
    h, w = image.shape[:2]
    ch = int(h * scale_factor)
    cw = int(w * scale_factor)
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    cropped = image[y0:y0+ch, x0:x0+cw]
    return _ensure_same_shape(image, cropped)


def contrast(image, gain):
    """Adjust contrast via gamma correction."""
    adjusted = ski.exposure.adjust_gamma(image, gain) # type: ignore -- it's a float input!
    return _ensure_same_shape(image, adjusted)


def luminance(image, factor):
    """Linearly scale luminance."""
    brightened = image * factor
    return _ensure_same_shape(image, brightened)


def blur(image, sigma):
    """Apply Gaussian blur."""
    blurred = ski.filters.gaussian(image, sigma=sigma, channel_axis=-1)
    return _ensure_same_shape(image, blurred)


def shear(image, shear_angle):
    """ Applies a perspective/shear transform to the image along the x-axis while keeping the entire image in view."""

    # Image dimensions
    H, W = image.shape[:2]

    # Create shear transform
    shear_tform = ski.transform.AffineTransform(shear=shear_angle)

    # Original image corners
    corners = np.array([
        [0, 0],
        [W, 0],
        [W, H],
        [0, H]
    ])

    # Transform corners
    transformed_corners = shear_tform(corners)

    # Find min and max coordinates to determine output size
    min_coords = transformed_corners.min(axis=0)
    max_coords = transformed_corners.max(axis=0)

    output_shape = np.ceil(max_coords - min_coords).astype(int)

    # Compute translation to shift image so it's fully in view
    translation = ski.transform.AffineTransform(translation=-min_coords)
    full_tform = shear_tform + translation  # compose shear + translation

    # Apply warp with the new output shape
    transformed_image = ski.transform.warp(image,
                                       inverse_map=full_tform.inverse,
                                       output_shape=(output_shape[1], output_shape[0]),
                                       mode='constant',
                                       cval=1)  # note: shape is (rows, cols)

    return _ensure_same_shape(image, transformed_image)


def process_one_image(image_path: Path, out_dir="transformed"):
    """
    Load an image, apply several augmentations using skimage, and save all variants.
    """

    # Load the input image
    img = u.load_image(image_path)

    # Generate random values within specified ranges
    rotation_angle = np.random.choice(
        [np.random.uniform(5, 18), 
         np.random.uniform(-5, -18),
         np.random.uniform(95, 118), 
         np.random.uniform(-95, -118)],
         )
    scale_factor = np.random.uniform(0.7, 0.9)
    contrast_factor = np.random.uniform(1.3, 1.8)
    luminance_factor = np.random.uniform(1.3, 1.8)
    blur_sigma = np.random.uniform(0.8, 1.2)
    shear_angle = (np.deg2rad(np.random.choice(
        [np.random.uniform(5, 20), 
         np.random.uniform(-5, -20)])),
         np.deg2rad(np.random.choice(
        [np.random.uniform(5, 20), 
         np.random.uniform(-5, -20)])))

    # Apply augmentations
    augmented = [
        ("original", img),
        ("rotate", rotate(img, rotation_angle)),
        ("rescale", rescale(img, scale_factor)),
        ("contrast", contrast(img, contrast_factor)),  # type: ignore - gamma <1 => brighter
        ("luminance", luminance(img, luminance_factor)), # type: ignore
        ("blur", blur(img, blur_sigma)),
        ("shear", shear(img, shear_angle)),
    ]

    # Save all augmented variants
    for suffix, img_aug in augmented:
        out_name = f"{image_path.stem}_{suffix}{image_path.suffix}"
        # print(f"saving ... {out_name}")

        u.save_image(img_aug, u.gen_path(image_path, out_name, out_dir))
        # print("...saved", out_dir, out_name)


def main():
    try:
        parser = argparse.ArgumentParser(
            description=(
                "Image Augmentation Tool\n"
                "\n"
                "This script generates augmented versions of input images using various transformations:\n"
                "  • Rotation\n"
                "  • Rescaling\n"
                "  • Contrast adjustment\n"
                "  • Luminance enhancement\n"
                "  • Gaussian blur\n"
                "  • Shear (perspective) transform\n"
                "\n"
                "You can provide a single image file or a directory containing multiple images.\n"
                "All augmented images are saved to the specified output directory."
            ),
            formatter_class=argparse.RawTextHelpFormatter
        )

        parser.add_argument(
            "path",
            type=str,
            help=(
                "Path to an image file or a directory containing images.\n"
                "Supported formats are determined by Utils.get_all_images()."
            )
        )

        parser.add_argument(
            "--output",
            type=str,
            default="transformed",
            help=(
                "Output directory to save augmented images.\n"
                "Defaults to './transformed' if not specified."
            )
        )

        args = parser.parse_args()

        np.random.seed(42)

        all_image_paths = []
        if os.path.isdir(args.path):
            all_image_paths = u.get_all_images(args.path)
        else:
            all_image_paths = [Path(args.path)]

        u.parallel_process(all_image_paths, (lambda img : process_one_image(img, args.output)), n_jobs=8)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

