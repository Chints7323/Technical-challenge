import numpy as np
import os
import cv2

def extract_images_from_npz(npz_file, output_dir, prefix='img'):
    data = np.load(npz_file)
    images = data['images']
    os.makedirs(output_dir, exist_ok=True)

    for idx, image in enumerate(images):
        if idx % 25 == 0:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            filename = os.path.join(output_dir, f"{prefix}_{idx+1:04d}.jpg")
            cv2.imwrite(filename, image_bgr)

base_npz_file = './base.npz'

# Define output directories
base_output_dir = 'base_images'

# Extract images from npz files
extract_images_from_npz(base_npz_file, base_output_dir, prefix='base')

print(f"Base training images saved to: {base_output_dir}")


# Script to split the data and create a yolo acceptable format directory.

# import os
# import shutil
# import random

# # Paths to your images and annotations
# images_dir = 'base_images'
# annotations_dir = 'base_images_labels'
# output_dir = 'yolo_custom_dataset'

# # Create output directories
# train_images_dir = os.path.join(output_dir, 'images/train')
# val_images_dir = os.path.join(output_dir, 'images/val')
# train_labels_dir = os.path.join(output_dir, 'labels/train')
# val_labels_dir = os.path.join(output_dir, 'labels/val')

# os.makedirs(train_images_dir, exist_ok=True)
# os.makedirs(val_images_dir, exist_ok=True)
# os.makedirs(train_labels_dir, exist_ok=True)
# os.makedirs(val_labels_dir, exist_ok=True)

# # Copy the class.txt file
# class_file_src = os.path.join(annotations_dir, 'class.txt')
# class_file_dst = os.path.join(output_dir, 'class.txt')
# if os.path.exists(class_file_src):
#     shutil.copy(class_file_src, class_file_dst)
# else:
#     print(f"Warning: {class_file_src} does not exist!")

# # Get list of images and annotations
# image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')])
# annotation_files = sorted([f for f in os.listdir(annotations_dir) if f.endswith('.txt') and f != 'class.txt'])

# # Ensure the lists are aligned
# image_files = [f for f in image_files if f.replace('.jpg', '.txt').replace('.png', '.txt') in annotation_files]

# # Shuffle the dataset
# random.seed(42)  # For reproducibility
# combined = list(zip(image_files, annotation_files))
# random.shuffle(combined)
# image_files[:], annotation_files[:] = zip(*combined)

# # Split the dataset (80% train, 20% val)
# split_idx = int(len(image_files) * 0.8)
# train_image_files = image_files[:split_idx]
# val_image_files = image_files[split_idx:]
# train_annotation_files = annotation_files[:split_idx]
# val_annotation_files = annotation_files[split_idx:]

# # Move the files to the respective directories
# for img_file, ann_file in zip(train_image_files, train_annotation_files):
#     shutil.copy(os.path.join(images_dir, img_file), os.path.join(train_images_dir, img_file))
#     shutil.copy(os.path.join(annotations_dir, ann_file), os.path.join(train_labels_dir, ann_file))

# for img_file, ann_file in zip(val_image_files, val_annotation_files):
#     shutil.copy(os.path.join(images_dir, img_file), os.path.join(val_images_dir, img_file))
#     shutil.copy(os.path.join(annotations_dir, ann_file), os.path.join(val_labels_dir, ann_file))

# print("Dataset split and organized successfully!")