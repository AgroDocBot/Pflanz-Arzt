import cv2
import numpy as np
import os

def generate_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25, 50, 50])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    return refined_mask

def separate_leaves(image, mask, save_path, filename_prefix="leaf_"):
    os.makedirs(save_path, exist_ok=True)

    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    sure_bg = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(image, markers)

    leaf_images = []
    for marker in range(2, markers.max() + 1):  
        leaf_mask = np.uint8(markers == marker) * 255
        x, y, w, h = cv2.boundingRect(leaf_mask)
        if w * h < 1000:  # ignore small regions
            continue

        cropped_leaf = cv2.bitwise_and(image, image, mask=leaf_mask)
        cropped_leaf = cropped_leaf[y:y+h, x:x+w]
        leaf_filename = os.path.join(save_path, f"{filename_prefix}{marker}.png")
        cv2.imwrite(leaf_filename, cropped_leaf)
        leaf_images.append(leaf_filename)
    return leaf_images

def segment_image(image_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[ERROR] Failed to load input image: {image_path}")

    mask = generate_mask(img)
    mask_path = os.path.join(output_dir, "binary_mask.png")
    cv2.imwrite(mask_path, mask)

    leaf_images = separate_leaves(img, mask, output_dir)

    if not leaf_images:
        raise RuntimeError("[ERROR] No leaves were segmented.")

    return leaf_images
