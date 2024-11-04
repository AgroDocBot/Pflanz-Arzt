import os
import cv2
from plantcv import plantcv as pcv

# Set input and output directories
input_dirs = {"train": "segmented"}
output_dirs = {"train": "segmented_processed"}

# Create output directories if they don't exist
for key, output_dir in output_dirs.items():
    os.makedirs(output_dir, exist_ok=True)

def preprocess_images(input_dir, output_dir):
    for label in os.listdir(input_dir):
        label_dir = os.path.join(input_dir, label)
        output_label_dir = os.path.join(output_dir, label)
        os.makedirs(output_label_dir, exist_ok=True)

        for image_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, image_name)
            img = cv2.imread(img_path)

            # Preprocess using PlantCV
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
            binary_img = pcv.threshold.otsu(blur_img)
            resized_img = cv2.resize(binary_img, (150, 150))

            processed_img_path = os.path.join(output_label_dir, image_name)
            cv2.imwrite(processed_img_path, resized_img)

# Preprocess images in both train and test directories
for key in input_dirs:
    preprocess_images(input_dirs[key], output_dirs[key])
