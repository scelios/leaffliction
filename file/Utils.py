#!/usr/bin/env python3

from ast import List
import os
import matplotlib

matplotlib.use('TkAgg')  # select a GUI backend BEFORE importing pyplot
from skimage import io, exposure
from skimage.util import img_as_ubyte
import numpy as np
import warnings


# suppress skimage low-contrast UserWarning
warnings.filterwarnings("ignore", message=".*low contrast.*")

def getNbImagesPerFolder(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")

    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    counts = {}
    
    for subdir in subdirs:
        subdir_path = os.path.join(directory, subdir)
        counts[subdir] = len([f for f in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, f))])

    # print("Number of images per folder:")
    # for subdir, count in counts.items():
    #     print(f"{subdir}: {count}")

    return counts

# helper to save with safe dtype and extension into the per-class output dir
def save_safe(img, dst_name, aug_subdir):
    base, ext = os.path.splitext(dst_name)
    if ext == '':
        ext = '.png'
        dst_name = base + ext
    dst_path = os.path.join(aug_subdir, dst_name)

    # boost contrast for very low-contrast images
    try:
        if exposure.is_low_contrast(img):
            img = exposure.equalize_adapthist(img)  # returns float in [0,1]
    except Exception:
        print(f"Warning: could not adjust contrast for image {dst_name}, saving original.")
        pass

    # convert floats to uint8 in [0..255]; handle floats not in [0,1]
    if np.issubdtype(img.dtype, np.floating):
        # if floats are not in [0,1], rescale from image range to [0,1]
        if img.max() > 1.0 or img.min() < 0.0:
            img = exposure.rescale_intensity(img, in_range='image', out_range=(0.0, 1.0))
        img = np.clip(img, 0.0, 1.0)
        img = img_as_ubyte(img)
    elif img.dtype != np.uint8:
        # convert other integer types to uint8
        img = img_as_ubyte(img)
    io.imsave(dst_path, img)


def transformDirectory(directory, pathTransformedImages, callback):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")

    print(f"Transforming images in directory: {directory}")

    os.makedirs(pathTransformedImages, exist_ok=True)

    # avoid processing the output folder if it's inside the source directory
    out_basename = os.path.basename(os.path.normpath(pathTransformedImages))
    for subdir in os.listdir(directory):
        if subdir == out_basename:
            continue
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            print(f"Processing subdirectory: {subdir}")

            # create per-class output folder
            aug_subdir = os.path.join(pathTransformedImages, subdir)
            os.makedirs(aug_subdir, exist_ok=True)

            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                if not os.path.isfile(file_path):
                    continue

                # read image
                image = io.imread(file_path)
                # continue
                # apply augmentations
                imags: List =  callback(image, filename, aug_subdir)
                # save image to disk

                for (im, name) in imags:
                    save_safe(im, name, aug_subdir)



def transformFile(file_path, filename, pathTransformedImages, callback):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    print(f"Transforming single image file: {file_path}")

    os.makedirs(pathTransformedImages, exist_ok=True)

    # read image
    image = io.imread(file_path)

    # save original copy
    save_safe(image, filename, pathTransformedImages)

    # apply augmentations
    imags: List =  callback(image, filename, pathTransformedImages)
    # save image to disk

    for (im, name) in imags:
        save_safe(im, name, pathTransformedImages)


