import os

# 输入文件夹路径和输出文件路径
image_dir = "/dataset/VisDrone2019-DET-train/VisDrone2019-DET-train/images/"
input_folder = 'VisDrone/train/annotations'
output_file = 'train_for_platform.txt'

# 遍历标注文件
annotations = {}
for file_name in os.listdir(input_folder):
    if file_name.endswith('.txt'):
        file_path = os.path.join(input_folder, file_name)
        with open(file_path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                # 解析每一行的锚框信息
                data = line.strip().split(',')
                bbox = [int(data[i]) for i in range(6)]

                # 如果置信度不为0，则保存锚框信息
                if bbox[4] != 0:
                    image_name = image_dir + file_name.split('.')[0] + '.jpg'

                    coordinates = [bbox[0], bbox[1], bbox[2]+bbox[0], bbox[3]+bbox[1]]
                    category = bbox[5] + 1

                    # 添加到字典中
                    if image_name in annotations:
                        annotations[image_name].append((coordinates, category))
                    else:
                        annotations[image_name] = [(coordinates, category)]

# 将结果写入输出文件
with open(output_file, 'w') as f:
    for image_name, bboxes in annotations.items():
        line = f'{image_name}'
        for bbox in bboxes:
            coordinates = bbox[0]
            category = bbox[1]
            line += f' {coordinates[0]},{coordinates[1]},{coordinates[2]},{coordinates[3]},{category}'
        line += '\n'
        f.write(line)