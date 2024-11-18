import os
import numpy as np
from PIL import Image

# Set file path
file_path = 'E:/download/OneDrive_2023-11-02/PHOTOVOLTAIC_THERMAL_IMAGES_DATASET/'
data_temp = np.load(os.path.join(file_path, 'imgs_temp.npy'))

print("ok1")
# Path to save adjusted images
output_folder = os.path.join(file_path, 'images')
os.makedirs(output_folder, exist_ok=True)  # Create a folder to save the images
print("ok2")
# Normalize the data to the range 0 to 255
data_temp = ((data_temp - np.min(data_temp)) / (np.max(data_temp) - np.min(data_temp))) * 255
print("ok3")
# Convert each array entry to an image and save as a JPG file
for i in range(len(data_temp)):
    img_array = data_temp[i]
    img_array = np.uint8(img_array)  # Convert to integer type
    img = Image.fromarray(img_array)
    img = img.convert('L')  # Convert to grayscale mode

    # Save the image as a JPG file to the specified folder
    img.save(os.path.join(output_folder, f'{i}.jpg'))
    print("ok4")
    # If needed, uncomment the next line to display the image during the conversion process
    # img.show()
