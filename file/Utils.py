#!/usr/bin/env python3

from ast import List
import os
from pathlib import Path
import matplotlib
matplotlib.use('TkAgg')  # select a GUI backend BEFORE importing pyplot

import skimage as ski
from skimage import io, exposure
from skimage.util import img_as_ubyte
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def load_image(path):
    """Load an image as float from path"""
    return ski.util.img_as_float(ski.io.imread(path))


def save_image(image, save_path: Path):
    """Save image of any type, (converted with img_as_ubyte)"""
    
    save_path.parent.mkdir(parents=True,  exist_ok=True)
    image = ski.util.img_as_ubyte(image)
    ski.io.imsave(save_path, image, check_contrast=False)


def gen_path(image_path: Path, filename_suffix: str, out_dir="transformed") -> Path:
    """
    Generate a new output path by merging the given image_path into out_dir.
    Keeps the last two directory levels of image_path, and replaces the filename
    with filename_suffix. Never raises errors for shallow paths.
    
    Example:
        image_path = Path("file/image/Apple/Apple_black_rot/image (1).jpg")
        filename_suffix = "rotated_image (1).jpg"
        out_dir = "transformed"
        -> transformed/Apple/Apple_black_rot/rotated_image (1).jpg
    """
    image_path = Path(image_path)
    out_dir = Path(out_dir)

    parts = image_path.parts
    # Get up to last 3 parts (2 dirs + filename)
    sub_parts = parts[-3:] if len(parts) >= 3 else parts
    # Replace filename with the provided suffix
    if sub_parts:
        sub_parts = list(sub_parts)
        sub_parts[-1] = filename_suffix

    new_subpath = Path(*sub_parts)
    return out_dir / new_subpath


def parallel_process(items, func, n_jobs=-1, use_tqdm=True):
    """Launch jobs in parallel with a tqdm progress bar"""
    if use_tqdm:
        return Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(func)(item) for item in tqdm(items)
        )
    else:
        return Parallel(n_jobs=n_jobs)(delayed(func)(item) for item in items)


def get_all_images(root_dir, exts=(".jpg", ".jpg", ".jpeg", ".png", ".tif", ".bmp")):
    root_dir = Path(root_dir)
    # Use rglob for recursive search; match extensions case-insensitively
    image_paths = [
        p for p in root_dir.rglob("*")
        if p.suffix.lower() in exts
    ]
    return sorted(image_paths)