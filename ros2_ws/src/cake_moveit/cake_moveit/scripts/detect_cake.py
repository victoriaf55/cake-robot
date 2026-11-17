"""Fit a circle or rectangle to a high-contrast cake in a saved image.

This is an offline first step: it reports image-pixel measurements only and
does not publish ROS messages or command the robot. It works best when the
cake is fully visible against a plain, contrasting background.

Example:
    python3 detect_cake.py --image cake.jpg
    python3 detect_cake.py --image cake.jpg --threshold 40 --show
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def corner_background(image):
    """Estimate the plain background color from four small image corners."""
    height, width = image.shape[:2]
    patch_h = max(3, int(height * 0.04))
    patch_w = max(3, int(width * 0.04))
    patches = (
        image[:patch_h, :patch_w],
        image[:patch_h, width - patch_w:],
        image[height - patch_h:, :patch_w],
        image[height - patch_h:, width - patch_w:],
    )
    pixels = np.concatenate([patch.reshape(-1, 3) for patch in patches], axis=0)
    return np.median(pixels, axis=0)


def make_foreground_mask(image, threshold, min_saturation,
                         smooth_kernel_size=0):
    """Segment cake-colored pixels while rejecting gray shadows on white."""
    background = corner_background(image).astype(np.float32)
    difference = image.astype(np.float32) - background
    distance = np.sqrt(np.sum(difference * difference, axis=2))

    # A cast shadow is usually darker than the white background but remains
    # nearly gray. A brown cookie cake is more saturated, so reject low-
    # saturation pixels that would otherwise extend the cake contour.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cake_color = ((distance >= threshold) &
                  (hsv[:, :, 1].astype(np.float32) >= min_saturation))
    mask = np.where(cake_color, 255, 0).astype(np.uint8)

    # Remove isolated specks and fill small gaps along the cake boundary.
    kernel_size = max(3, (min(image.shape[:2]) // 150) | 1)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Blur and re-threshold the shared mask to soften small edge irregularities
    # before either the circle or rectangle is classified and fitted.
    blur_size = (smooth_kernel_size or
                 max(3, (min(image.shape[:2]) // 200) | 1))
    smoothed = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    _, mask = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
    return mask


def find_cake(image, threshold=30.0, min_area_fraction=0.01,
              min_saturation=30.0, smooth_kernel_size=0):
    """Return the cake's fitted shape, pixel geometry, and foreground mask."""
    mask = make_foreground_mask(
        image, threshold, min_saturation, smooth_kernel_size)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_area = image.shape[0] * image.shape[1]
    contours = [c for c in contours
                if cv2.contourArea(c) >= image_area * min_area_fraction]
    if not contours:
        raise ValueError(
            "No large foreground contour found. Use a contrasting background "
            "or lower --threshold / --min-area-fraction.")

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    circularity = (4.0 * np.pi * area / (perimeter * perimeter)
                   if perimeter > 0 else 0.0)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) == 4 and cv2.isContourConvex(approx):
        shape = "rectangle"
        # Assume the cake edges are horizontal/vertical; fit an axis-aligned
        # box to the simplified contour rather than estimating rotation.
        x, y, width, height = cv2.boundingRect(approx)
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        fitted_contour = np.array(
            [[x, y], [x + width, y], [x + width, y + height],
             [x, y + height]], dtype=np.int32).reshape(-1, 1, 2)
        shape_area = float(width * height)
        geometry = {
            "width_px": float(width),
            "height_px": float(height),
        }
    elif circularity >= 0.82:
        shape = "circle"
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        fitted_contour = cv2.ellipse2Poly(
            (round(center_x), round(center_y)),
            (round(radius), round(radius)), 0, 0, 360, 5).reshape(-1, 1, 2)
        shape_area = float(np.pi * radius * radius)
        geometry = {"diameter_px": round(float(2.0 * radius), 1)}
    else:
        shape = "unknown"
        rotated = cv2.minAreaRect(contour)
        (center_x, center_y), (width, height), _ = rotated
        fitted_contour = contour
        shape_area = area
        geometry = {
            "width_px": round(float(width), 1),
            "height_px": round(float(height), 1),
        }

    result = {
        "shape": shape,
        "center_px": [round(float(center_x), 1), round(float(center_y), 1)],
        "contour_area_px2": round(area, 1),
        "shape_area_px2": round(shape_area, 1),
        "circularity": round(float(circularity), 3),
        **geometry,
    }
    return result, fitted_contour, mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True,
                        help="Path to an image file (JPG, PNG, etc.)")
    parser.add_argument("--threshold", type=float, default=30.0,
                        help="BGR color-distance threshold from the corners")
    parser.add_argument("--min-saturation", type=float, default=30.0,
                        help="Minimum HSV saturation to reject gray shadows "
                             "(0 disables this filter; range 0-255)")
    parser.add_argument("--smooth-kernel-size", type=int, default=0,
                        help="Gaussian mask-smoothing width in pixels; use an "
                             "odd value >= 3, or 0 for automatic sizing")
    parser.add_argument("--min-area-fraction", type=float, default=0.01,
                        help="Ignore contours smaller than this image fraction")
    parser.add_argument("--output", help="Annotated image output path")
    parser.add_argument("--show", action="store_true",
                        help="Open a window with the annotated image")
    args = parser.parse_args()

    if args.threshold < 0:
        parser.error("--threshold must be nonnegative")
    if not 0 <= args.min_saturation <= 255:
        parser.error("--min-saturation must be between 0 and 255")
    if args.smooth_kernel_size != 0 and (
            args.smooth_kernel_size < 3 or args.smooth_kernel_size % 2 == 0):
        parser.error("--smooth-kernel-size must be 0 or an odd integer >= 3")
    if not 0 < args.min_area_fraction < 1:
        parser.error("--min-area-fraction must be between 0 and 1")

    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        parser.error(f"Could not read image: {image_path}")

    try:
        detection, fitted_contour, mask = find_cake(
            image, threshold=args.threshold,
            min_area_fraction=args.min_area_fraction,
            min_saturation=args.min_saturation,
            smooth_kernel_size=args.smooth_kernel_size)
    except ValueError as exc:
        parser.error(str(exc))

    preview = image.copy()
    cv2.drawContours(preview, [fitted_contour], -1, (0, 220, 0), 3)
    center = tuple(int(round(value)) for value in detection["center_px"])
    cv2.drawMarker(preview, center, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    label = f"{detection['shape']} | center={detection['center_px']} px"
    cv2.putText(preview, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2, cv2.LINE_AA)

    output_path = (Path(args.output) if args.output else
                   image_path.with_name(image_path.stem + "_detected.png"))
    if not cv2.imwrite(str(output_path), preview):
        parser.error(f"Could not save preview: {output_path}")

    print(json.dumps(detection, indent=2))
    print(f"Annotated preview saved to: {output_path}")
    if args.show:
        cv2.imshow("Cake detection", preview)
        cv2.imshow("Foreground mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
