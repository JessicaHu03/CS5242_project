import os

# Input folder path and output file path
image_dir = "/dataset/VisDrone2019-DET-train/VisDrone2019-DET-train/images/"
input_folder = 'VisDrone/train/annotations'
output_file = 'train_for_platform.txt'

# Iterate through annotation files
annotations = {}
for file_name in os.listdir(input_folder):
    if file_name.endswith('.txt'):
        file_path = os.path.join(input_folder, file_name)
        with open(file_path, 'r') as f:
            lines = f.readlines()

            for line in lines:
                # Parse bounding box information from each line
                data = line.strip().split(',')
                bbox = [int(data[i]) for i in range(6)]

                # Save bounding box information if confidence is not zero
                if bbox[4] != 0:
                    image_name = image_dir + file_name.split('.')[0] + '.jpg'

                    coordinates = [bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]
                    category = bbox[5] + 1

                    # Add to the dictionary
                    if image_name in annotations:
                        annotations[image_name].append((coordinates, category))
                    else:
                        annotations[image_name] = [(coordinates, category)]

# Write the results to the output file
with open(output_file, 'w') as f:
    for image_name, bboxes in annotations.items():
        line = f'{image_name}'
        for bbox in bboxes:
            coordinates = bbox[0]
            category = bbox[1]
            line += f' {coordinates[0]},{coordinates[1]},{coordinates[2]},{coordinates[3]},{category}'
        line += '\n'
        f.write(line)
