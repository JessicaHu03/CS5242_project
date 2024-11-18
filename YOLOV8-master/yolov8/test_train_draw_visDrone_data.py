import cv2

# Read the image and get its dimensions
img_path = r"E:\data\EggsofAlive\JPEGImages/DJI0010.jpg"
img = cv2.imread(img_path)
height, width, _ = img.shape

# Parse all anchor box information
# Uncomment the loop if processing multiple lines in the file
# for i in range(2):
with open(r"test.txt") as f:
    line = f.readline().split(' ')

    file_path = line[0]
    anchor_boxes = []
    for item in line[1:]:
        bbox = [int(num) for num in item.split(',')]
        anchor_boxes.append(bbox)

# Draw rectangles for the anchor boxes
for box in anchor_boxes:
    x1, y1, x2, y2 = box[:4]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # Draw label text
    label_text = f"Label {box[-1]}"
    label_pos = (x1, y1 - 10)
    cv2.putText(img, label_text, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

# Save the image to a file
output_image_path = "output_image.jpg"
cv2.imwrite(output_image_path, img)

print(f"Image saved as {output_image_path}")
