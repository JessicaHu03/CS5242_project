import os
import random
import shutil

# Path to the dataset directory
dataset_dir = r'C:\Users\Lenovo\Desktop\PHOTOVOLTAIC_THERMAL_IMAGES_DATASET\JPEGImages'

# Retrieve all image files
all_images = os.listdir(dataset_dir)
random.shuffle(all_images)

# Calculate the number of images for training, validation, and testing
total_images = len(all_images)
train_count = int(0.7 * total_images)  # 70% for training
val_count = int(0.2 * total_images)   # 20% for validation
test_count = total_images - train_count - val_count  # Remaining 10% for testing

# Split the dataset
train_images = all_images[:train_count]
val_images = all_images[train_count:train_count + val_count]
test_images = all_images[train_count + val_count:]

# Create directories for train, val, and test sets
train_path = os.path.join(dataset_dir, 'train')
val_path = os.path.join(dataset_dir, 'val')
test_path = os.path.join(dataset_dir, 'test')

os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Move images to their respective folders
for img in train_images:
    shutil.move(os.path.join(dataset_dir, img), os.path.join(train_path, img))

for img in val_images:
    shutil.move(os.path.join(dataset_dir, img), os.path.join(val_path, img))

for img in test_images:
    shutil.move(os.path.join(dataset_dir, img), os.path.join(test_path, img))
