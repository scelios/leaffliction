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

img, path, filename = pcv.readimage("file/images/Apple/Apple_scab/image (617).JPG") # type: ignore - it works!
print(path, filename)

colorspace_img = pcv.visualize.colorspaces(rgb_img=img)

b_img = pcv.rgb2gray_lab(rgb_img=img, channel='b')

hist_figure1, hist_data1 = pcv.visualize.histogram(img = b_img, hist_data=True)


thresh_mask = pcv.threshold.binary(gray_img=b_img, threshold=130, object_type='light')

fill_mask = pcv.fill(bin_img=thresh_mask, size=1000)

# pcv.plot_image(fill_mask)

analysis_image = pcv.analyze.size(img=img, labeled_mask=fill_mask)

# pcv.plot_image(analysis_image)
# pcv.plot_image(thresh_mask)

comp = pcv.visualize.tile([img, analysis_image], ncol=2)

# pcv.plot_image(comp)


# color_histogram = pcv.analyze.color(rgb_img=b_img, labeled_mask=fill_mask, colorspaces='all', label="default")

# pcv.plot_image(color_histogram)

pcv.outputs.save_results(filename="foobar")



def original(img):
    return img


def gaussian_blur(img):
    b_img = pcv.rgb2gray(rgb_img=img)
    detail_threashhold = pcv.threshold.binary(gray_img=b_img, threshold=120, object_type='light')
    b_img = pcv.rgb2gray_lab(rgb_img=img, channel='b')
    shape_mask = pcv.threshold.binary(gray_img=b_img, threshold=130, object_type='light')

    composed_img = pcv.apply_mask(img=detail_threashhold, mask=shape_mask, mask_color='black')

    gaussian = pcv.gaussian_blur(composed_img, ksize=(3, 3), sigma_x=0, sigma_y=None)
    return gaussian


def mask(img):
    b_img = pcv.rgb2gray_lab(rgb_img=img, channel='b')
    thresh_mask = pcv.threshold.binary(gray_img=b_img, threshold=130, object_type='light')
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
        cv2.drawContours(img_out, cnt, -1, color=(0, 255, 0), thickness=2)  # Green contours

        # all_points = np.vstack([cnt.reshape(-1, 2) for cnt in roi.contours])
        all_points = np.vstack(cnt[0])
        x_min = np.min(all_points[:, 0])
        x_max = np.max(all_points[:, 0])
        y_min = np.min(all_points[:, 1])
        y_max = np.max(all_points[:, 1])
        cv2.rectangle(img_out, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=2)  # Blue rectangle
    return img_out


def analyze_object(img):
    b_img = pcv.rgb2gray_lab(rgb_img=img, channel='b')
    thresh_mask = pcv.threshold.binary(gray_img=b_img, threshold=130, object_type='light')
    fill_mask = pcv.fill(bin_img=thresh_mask, size=1000)
    analysis_image = pcv.analyze.size(img=img, labeled_mask=fill_mask)
    return analysis_image


def pseudolandmarks(img):
    # Identify a set of land mark points
    # Results in set of point values that may indicate tip points
    left, right, center_h  = pcv.homology.y_axis_pseudolandmarks(img=img, mask=mask(img))

    # Plot points on image
    img_out = img.copy()
    def plot_points(points, color):
        for pt in points:
            cv2.circle(img_out, (int(pt[0][0]), int(pt[0][1])), radius=3, color=color, thickness=-1)

    plot_points(left, (255, 0, 0))       # Red for left
    plot_points(right, (0, 255, 0))      # Green for right
    plot_points(center_h, (0, 0, 255))   # Blue for center

    return img_out


def color_histogram(img):
    # plantCV color_histogram
    hist_figure1, hist_data1 = pcv.visualize.histogram(img = img, hist_data=True)
    return hist_data1


def main():
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
    fig = plt.figure(figsize=(10, 20), constrained_layout=True)
    gs = GridSpec(4, 2, figure=fig)  # 8 rows, 2 cols

    # Map first 6 functions to grid positions
    positions = [(0,0),(0,1),(1,0),(1,1),(2,0),(2,1)]
    for (func, title), (r, c) in zip(functions, positions):
        ax = fig.add_subplot(gs[r, c])
        ax.set_title(title)
        if title == "Gaussian blur":
            ax.imshow(func(img), cmap='gray')
        else:
            ax.imshow(func(img))

    # Last function (histogram) takes 2 columns
    # ax_hist = fig.add_subplot(gs[3, :])  # spans row 3-4, all columns
    # # hist_data = color_histogram(img)
    # ax_hist.bar(range(len(hist_data)), hist_data)
    # ax_hist.set_title("Color histogram")


    # plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    img, path, filename = pcv.readimage("file/images/Apple/Apple_scab/image (617).JPG") # type: ignore - it works!

    main()