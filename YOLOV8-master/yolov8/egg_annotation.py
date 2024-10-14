import os
import xml.etree.ElementTree as ET

# 定义VOC标签到类别名称的映射（根据您的数据集修改）
class_names = {
    "0": "eggs",
    # 添加更多类别映射
}

# 输入XML文件所在的目录
xml_dir = "E:\data\EggsofAlive\Annotations"

# 输出txt文件的路径
output_txt_path = "egg_train.txt"

# 打开txt文件以写入内容
with open(output_txt_path, "w") as txt_file:
    # 遍历XML文件目录中的所有XML文件
    for filename in os.listdir(xml_dir):
        if filename.endswith(".xml"):
            xml_file_path = os.path.join(xml_dir, filename)

            # 解析XML文件
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            # 遍历XML中的所有标注对象
            for obj in root.findall("object"):
                # 获取类别信息
                class_id = obj.find("name").text
                class_label = class_names.get(class_id, "0")

                # 获取坐标信息
                bbox = obj.find("bndbox")
                xmin = bbox.find("xmin").text
                ymin = bbox.find("ymin").text
                xmax = bbox.find("xmax").text
                ymax = bbox.find("ymax").text

                # 构建每一行的格式
                line = f"{filename} {xmin},{ymin},{xmax},{ymax},{class_label}\n"

                # 写入txt文件
                txt_file.write(line)

print(f"转换完成，输出文件：{output_txt_path}")
