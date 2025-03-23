import os
import cv2

# Correctly define the input and output directories using raw strings
input_dir = r'C:\Users\hp\OneDrive\Documents\segmentationdataset\mask'
output_dir = r'C:\Users\hp\OneDrive\Documents\segmentationdataset\image-segmentation-yolov8\labels'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Iterate through all files in the input directory
for j in os.listdir(input_dir):
    image_path = os.path.join(input_dir, j)

    # Load the binary mask and get its contours
    mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Skipping {image_path}: file could not be read as an image.")
        continue

    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

    H, W = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Convert the contours to polygons
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 200:
            polygon = []
            for point in cnt:
                x, y = point[0]
                polygon.append(x / W)
                polygon.append(y / H)
            polygons.append(polygon)

    # Write polygons to a text file
    output_file = os.path.join(output_dir, f"{os.path.splitext(j)[0]}.txt")
    with open(output_file, 'w') as f:
        for polygon in polygons:
            for p_, p in enumerate(polygon):
                if p_ == 0:
                    f.write(f"0 {p} ")
                elif p_ == len(polygon) - 1:
                    f.write(f"{p}\n")
                else:
                    f.write(f"{p} ")
