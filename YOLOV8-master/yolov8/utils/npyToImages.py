import os
import numpy as np
from PIL import Image

# 设定文件路径
file_path = 'E:/download/OneDrive_2023-11-02/PHOTOVOLTAIC_THERMAL_IMAGES_DATASET/'
data_temp = np.load(os.path.join(file_path, 'imgs_temp.npy'))

print("ok1")
# 调整并保存图像的路径
output_folder = os.path.join(file_path, 'images')
os.makedirs(output_folder, exist_ok=True)  # 创建保存图像的文件夹
print("ok2")
# 将数据规范化到 0 到 255 之间
data_temp = ((data_temp - np.min(data_temp)) / (np.max(data_temp) - np.min(data_temp))) * 255
print("ok3")
# 将每个数组项转换为图像并保存为 jpg 文件
for i in range(len(data_temp)):
    img_array = data_temp[i]
    img_array = np.uint8(img_array)  # 转换为整数类型
    img = Image.fromarray(img_array)
    img = img.convert('L')  # 转换为灰度图像模式

    # 保存图像为 jpg 文件到指定文件夹
    img.save(os.path.join(output_folder, f'{i}.jpg'))
    print("ok4")
    # 如果需要，您可以取消下一行注释来在转换过程中显示图像
    # img.show()


