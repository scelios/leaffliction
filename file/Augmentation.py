import os
import sys
import argparse
import matplotlib
matplotlib.use('TkAgg')  # select a GUI backend BEFORE importing pyplot
import matplotlib.pyplot as plt
from skimage import io, exposure
from skimage.util import img_as_ubyte
from skimage.transform import rotate, AffineTransform, warp, rescale, resize, downscale_local_mean, ProjectiveTransform
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

# skew: simulate a 3D camera movement via a projective (perspective) transform
def perspective_transform(img, max_shift=0.15):
    h, w = img.shape[:2]
    src = np.array([[0, 0],
                    [w, 0],
                    [w, h],
                    [0, h]])
    # random shifts as fraction of width/height
    sx = np.random.uniform(-max_shift, max_shift)
    sy = np.random.uniform(-max_shift, max_shift)

    dst = src.astype(np.float32).copy()
    # move top corners more to emulate camera tilt/shift; bottom corners less
    dst[0] += [ sx * w,  sy * h]   # top-left
    dst[1] += [-sx * w,  sy * h]   # top-right
    dst[2] += [-0.3*sx * w, -0.3*sy * h]  # bottom-right
    dst[3] += [ 0.3*sx * w, -0.3*sy * h]  # bottom-left

    tform = ProjectiveTransform()
    if not tform.estimate(src, dst):
        return img
    warped = warp(img, tform, output_shape=(h, w), preserve_range=True)
    return warped


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

def augment(directory, counts, PathAugmentedImages):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")

    print(f"Augmenting images in directory: {directory}")

    os.makedirs(PathAugmentedImages, exist_ok=True)

    # avoid processing the output folder if it's inside the source directory
    out_basename = os.path.basename(os.path.normpath(PathAugmentedImages))

    for subdir in os.listdir(directory):
        if subdir == out_basename:
            continue
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            print(f"Processing subdirectory: {subdir}")

            # create per-class output folder
            aug_subdir = os.path.join(PathAugmentedImages, subdir)
            os.makedirs(aug_subdir, exist_ok=True)

            for filename in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, filename)
                if not os.path.isfile(file_path):
                    continue

                # if the folder has at least the same number of images than the higher count of the original breakdown
                if counts[subdir] >= max(counts.values()):
                    print(f"  Skipping further augmentation for class '{subdir}' (has {counts[subdir]} images).")
                    break
                
                # read image
                image = io.imread(file_path)


                name_base = os.path.splitext(filename)[0]
                name_ext = os.path.splitext(filename)[1] or '.png'

                # flip
                flipped = image[:, ::-1]
                save_safe(flipped, f"{name_base}_Flip{name_ext}", aug_subdir)

                # skew
                skewed = warp(image, AffineTransform(shear=0.2))
                save_safe(skewed, f"{name_base}_Skew{name_ext}", aug_subdir)

                # shear
                sheared = warp(image, AffineTransform(shear=0.3))
                save_safe(sheared, f"{name_base}_Shear{name_ext}", aug_subdir)

                # rotate but keep full image (expand canvas)
                angle = np.random.uniform(-30, 30)  # random angle in degrees
                rotated = rotate(image, angle=angle, resize=True, preserve_range=True)
                save_safe(rotated, f"{name_base}_Rot{name_ext}", aug_subdir)

                
                skewed = perspective_transform(image, max_shift=0.15)
                save_safe(skewed, f"{name_base}_Skew{name_ext}", aug_subdir)

                # crop: reduce image to 80% of original (i.e. crop 20%) centered
                h, w = image.shape[:2]
                ch = int(h * 0.8)
                cw = int(w * 0.8)
                y0 = max(0, (h - ch) // 2)
                x0 = max(0, (w - cw) // 2)
                cropped = image[y0:y0+ch, x0:x0+cw]
                save_safe(cropped, f"{name_base}_Crop{name_ext}", aug_subdir)

                # distortion / noise
                distorted = image.astype(np.float64) + 0.5 * image.std() * np.random.random(image.shape)
                save_safe(distorted, f"{name_base}_Distortion{name_ext}", aug_subdir)

                # save original copy
                save_safe(image, filename, aug_subdir)
                
def main():
    try:
        parser = argparse.ArgumentParser(description="Create new images using data augmentation techniques \n• Flip\n• Rotate\n• Skew\n• Contrast\n• Crop\n• Distortion")
        parser.add_argument("path", type=str, help="Path to the Images directory")
        parser.add_argument("--output", type=str, default=None, help="Path to save augmented images (default: <path>/augmented/)")
        args = parser.parse_args()

        # if --output not provided, use <path>/augmented
        output_dir = args.output if args.output else os.path.join(args.path, "augmented")
        imagesPerFolder = getNbImagesPerFolder(args.path)

        augment(args.path, imagesPerFolder, output_dir)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

