import os
import xml.etree.ElementTree as ET

# Define the mapping from VOC labels to class names (modify according to your dataset)
class_names = {
    "0": "eggs",
    # Add more class mappings
}

# Directory containing the input XML files
xml_dir = "E:\data\EggsofAlive\Annotations"

# Path for the output txt file
output_txt_path = "egg_train.txt"

# Open the txt file to write content
with open(output_txt_path, "w") as txt_file:
    # Iterate through all XML files in the directory
    for filename in os.listdir(xml_dir):
        if filename.endswith(".xml"):
            xml_file_path = os.path.join(xml_dir, filename)

            # Parse the XML file
            tree = ET.parse(xml_file_path)
            root = tree.getroot()

            # Iterate through all annotated objects in the XML
            for obj in root.findall("object"):
                # Get the class information
                class_id = obj.find("name").text
                class_label = class_names.get(class_id, "0")

                # Get the bounding box coordinates
                bbox = obj.find("bndbox")
                xmin = bbox.find("xmin").text
                ymin = bbox.find("ymin").text
                xmax = bbox.find("xmax").text
                ymax = bbox.find("ymax").text

                # Construct the format for each line
                line = f"{filename} {xmin},{ymin},{xmax},{ymax},{class_label}\n"

                # Write to the txt file
                txt_file.write(line)

print(f"Conversion complete. Output file: {output_txt_path}")
