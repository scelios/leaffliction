import argparse
import sys
from matplotlib.gridspec import GridSpec
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Set debug to the global parameter
# pcv.params.debug = "plot"
# Change display settings
# pcv.params.dpi = 100
# Increase text size and thickness to make labels clearer
# (size may need to be altered based on original image size)
pcv.params.text_size = 5
pcv.params.text_thickness = 10
pcv.params.line_thickness = 2


def original(img):
    return img


def gaussian_blur(img):
    b_img = pcv.rgb2gray(rgb_img=img)
    detail_threashhold = pcv.threshold.binary(
        gray_img=b_img, threshold=120, object_type='light')
    b_img = pcv.rgb2gray_lab(rgb_img=img, channel='b')
    shape_mask = pcv.threshold.binary(
        gray_img=b_img, threshold=130, object_type='light')

    composed_img = pcv.apply_mask(
        img=detail_threashhold, mask=shape_mask, mask_color='black')

    gaussian = pcv.gaussian_blur(
        composed_img, ksize=(3, 3), sigma_x=0, sigma_y=None)
    return gaussian


def mask(img):
    b_img = pcv.rgb2gray_lab(rgb_img=img, channel='b')
    thresh_mask = pcv.threshold.binary(
        gray_img=b_img, threshold=130, object_type='light')
    fill_mask = pcv.fill(bin_img=thresh_mask, size=1000)
    return fill_mask


def img_mask(img):
    b_img = gaussian_blur(img)
    composed_img = pcv.apply_mask(img=img, mask=b_img, mask_color='white')
    return composed_img


def roi_objects(img):
    roi = pcv.roi.from_binary_image(img, mask(img))

    # Plot each contour
    img_out = img.copy()
    for cnt in roi.contours:
        cv2.drawContours(img_out, cnt, -1, color=(0, 255, 0),
                         thickness=2)  # Green contours

        # all_points = np.vstack([cnt.reshape(-1, 2) for cnt in roi.contours])
        all_points = np.vstack(cnt[0])
        x_min = np.min(all_points[:, 0])
        x_max = np.max(all_points[:, 0])
        y_min = np.min(all_points[:, 1])
        y_max = np.max(all_points[:, 1])
        cv2.rectangle(img_out, (x_min, y_min), (x_max, y_max),
                      color=(0, 0, 255), thickness=2)  # Blue rectangle
    return img_out


def analyze_object(img):
    b_img = pcv.rgb2gray_lab(rgb_img=img, channel='b')
    thresh_mask = pcv.threshold.binary(
        gray_img=b_img, threshold=130, object_type='light')
    fill_mask = pcv.fill(bin_img=thresh_mask, size=1000)
    analysis_image = pcv.analyze.size(img=img, labeled_mask=fill_mask)
    return analysis_image


def pseudolandmarks(img):
    # Identify a set of land mark points
    # Results in set of point values that may indicate tip points
    left, right, center_h = pcv.homology.y_axis_pseudolandmarks(
        img=img, mask=mask(img))

    # Plot points on image
    img_out = img.copy()

    def plot_points(points, color):
        for pt in points:
            cv2.circle(img_out, (int(pt[0][0]), int(
                pt[0][1])), radius=3, color=color, thickness=-1)

    plot_points(left, (255, 0, 0))       # Red for left
    plot_points(right, (0, 255, 0))      # Green for right
    plot_points(center_h, (0, 0, 255))   # Blue for center

    return img_out


def color_histogram(img):
    # plantCV color_histogram
    hist_figure1, hist_data1 = pcv.visualize.histogram(img=img, hist_data=True)
    return hist_data1


def main(img):
    # Plot multiple images in one figure using matplotlib
    functions = [
        (original, "Original"),
        (gaussian_blur, "Gaussian blur"),
        (img_mask, "Mask"),
        (roi_objects, "ROI objects"),
        (analyze_object, "Analyze object"),
        (pseudolandmarks, "Pseudolandmarks"),
    ]

    # Create GridSpec
    fig = plt.figure(figsize=(6, 10), constrained_layout=True)
    gs = GridSpec(4, 2, figure=fig)  # 8 rows, 2 cols

    # Map first 6 functions to grid positions
    positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    for (func, title), (r, c) in zip(functions, positions):
        ax = fig.add_subplot(gs[r, c])
        ax.set_title(title)
        if title == "Gaussian blur":
            ax.imshow(func(img), cmap='gray')
        else:
            ax.imshow(func(img))

    # Histogram (takes 2 columns)
    ax_hist = fig.add_subplot(gs[3, :])  # spans row 3, all columns

    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Extract channels
    b_channel = img[:, :, 2]    # Blue
    g_channel = img[:, :, 1]    # Green
    r_channel = img[:, :, 0]    # Red

    # LAB channels
    lab_l = lab_img[:, :, 0]
    lab_a = lab_img[:, :, 1]  # Green-Magenta
    lab_b = lab_img[:, :, 2]  # Blue-Yellow

    # HSV channels
    hue = hsv_img[:, :, 0]
    sat = hsv_img[:, :, 1]
    val = hsv_img[:, :, 2]

    # Normalize histograms
    channels = {
        "Blue": b_channel,
        "Blue-Yellow": lab_b,
        "Green": g_channel,
        "Green-Magenta": lab_a,
        "Hue": hue,
        "Lightness": lab_l,
        "Red": r_channel,
        "Saturation": sat,
        "Value": val
    }

    for label, channel in channels.items():
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = hist * 100
        ax_hist.plot(hist, label=label, linewidth=1.5)

    ax_hist.set_title("Color and Intensity Histograms")
    ax_hist.set_xlabel("Pixel Intensity")
    ax_hist.set_ylabel("Proportion of Pixels (%)")
    ax_hist.legend(loc="upper right", fontsize=8)

    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description=(
                "Image Transformation Tool\n"
            ),
            formatter_class=argparse.RawTextHelpFormatter
        )

        parser.add_argument(
            "path",
            type=str,
            help=(
                "Path to an image file or a directory containing images.\n"
            )
        )

        args = parser.parse_args()

        img, path, filename = pcv.readimage(
            args.path)  # type: ignore - it works!
        main(img)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
