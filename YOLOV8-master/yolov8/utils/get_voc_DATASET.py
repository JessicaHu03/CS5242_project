import os
import random
import shutil

dataset_dir = r'C:\Users\Lenovo\Desktop\PHOTOVOLTAIC_THERMAL_IMAGES_DATASET\JPEGImages'

# 获取所有图像文件
all_images = os.listdir(dataset_dir)
random.shuffle(all_images)

total_images = len(all_images)
train_count = int(0.7 * total_images)
val_count = int(0.2 * total_images)
test_count = total_images - train_count - val_count

train_images = all_images[:train_count]
val_images = all_images[train_count:train_count + val_count]
test_images = all_images[train_count + val_count:]

# 创建train、val和test文件夹
train_path = os.path.join(dataset_dir, 'train')
val_path = os.path.join(dataset_dir, 'val')
test_path = os.path.join(dataset_dir, 'test')

os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# 移动文件到相应文件夹
for img in train_images:
    shutil.move(os.path.join(dataset_dir, img), os.path.join(train_path, img))

for img in val_images:
    shutil.move(os.path.join(dataset_dir, img), os.path.join(val_path, img))

for img in test_images:
    shutil.move(os.path.join(dataset_dir, img), os.path.join(test_path, img))
