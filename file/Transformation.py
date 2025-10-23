#!/usr/bin/env python3

from ast import List
import os
import sys
import argparse
import matplotlib

matplotlib.use('TkAgg')  # select a GUI backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from skimage import color, filters, morphology, measure, segmentation
from skimage.draw import rectangle_perimeter
import numpy as np
import Utils

def modifyImage(image, filename, aug_subdir):
    imageSet = []

    name_base = os.path.splitext(filename)[0]
    name_ext = os.path.splitext(filename)[1] or '.png'


    # Convert to grayscale
    gray_img = color.rgb2gray(image)

    # blur using Gaussian filter
    gray_img = filters.gaussian(gray_img, sigma=1, preserve_range=True)



    # local threshold to create binary mask
    # thresh = filters.threshold_local(gray_img, block_size=51)

    # global threshold using Otsu's method
    thresh = filters.threshold_otsu(gray_img)
    binary_mask = gray_img < thresh

    # clean up small artifacts in the binary mask
    binary_mask = morphology.remove_small_objects(binary_mask, min_size=10)
    binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=20)

    # ensure boolean mask
    binary_mask = binary_mask.astype(bool)
    binary_mask = filters.gaussian(binary_mask, sigma=1, preserve_range=True)
    # save binary mask
    imageSet.append((binary_mask, f"{name_base}_BinaryMask{name_ext}"))

    # overwrite original image with masked version (keep dtype)
    originalImage = image.copy()
    image = (image * binary_mask[:, :, np.newaxis]).astype(image.dtype).copy()

    # clean up small artifacts in the masked image
    # Convert to grayscale for morphology operations

    # save masked (foreground) image — use safe RGB indexing
    masked_image = image.copy()
    masked_image[~binary_mask, :] = 0
    imageSet.append((masked_image, f"{name_base}_Foreground{name_ext}"))

    # annotate the preserved original: paint red where the mask is True (white on masked image)
    annotated_original = originalImage.copy()
    # boolean mask is 2D; use it to index rows in the color image
    annotated_original[binary_mask, :] = [0, 255, 0]
    imageSet.append((annotated_original, f"{name_base}_ROI{name_ext}"))


    # compute mask boundary and draw a blue line on the original (edge between white and black)
    edge = segmentation.find_boundaries(binary_mask, mode='inner')
    thickness_radius = 2
    selem = morphology.disk(thickness_radius)
    thick_edge = morphology.binary_dilation(edge, selem)
    blue_edge_image = originalImage.copy()
    blue_edge_image[thick_edge, :] = [0, 0, 255]
    imageSet.append((blue_edge_image, f"{name_base}_Analyze{name_ext}"))

    




    # local treshold again to refine mask
    # masked_image = masked_image.rgb2gray().threshold_otsu()

    return imageSet


def main():
    try:
        parser = argparse.ArgumentParser(description="Transform images using various techniques \n• Gaussian Blur\n• Mask\n• Roi object\n• Analyze object\n• Pseudolandmarks")
        parser.add_argument("path", type=str, help="Path to the Images or directory")
        parser.add_argument("--output", type=str, default=None, help="Path to save transformed images (default: <path>/transformed/)")
        args = parser.parse_args()


        # if it is a directory
        if os.path.isdir(args.path):
            # if --output not provided, use <path>/transformed
            output_dir = args.output if args.output else os.path.join(args.path, "transformed")
            Utils.transformDirectory(args.path, output_dir, modifyImage)
        else:
            # if --output not provided, use <parent_of_path>/transformed
            output_dir = args.output if args.output else os.path.join(os.path.dirname(args.path), "transformed")
            Utils.transformFile(args.path, os.path.basename(args.path), output_dir, modifyImage)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

