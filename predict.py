from ultralytics import YOLO
import cv2
import numpy as np

# Define paths to the model and the image
model_path = r'C:\Users\hp\OneDrive\Documents\segmentationdataset\image-segmentation-yolov8\resultofprediction.pt'
image_path = r'C:\Users\hp\OneDrive\Documents\segmentationdataset\image-segmentation-yolov8\images\val\img_130.jpg'

# Read the input image
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

H, W, _ = img.shape

# Load the YOLO model
model = YOLO(model_path)

# Perform inference
results = model(img)

# Process the results
for i, result in enumerate(results):
    for j, mask in enumerate(result.masks.data):
        # Convert mask to a binary image
        mask = (mask.numpy() * 255).astype(np.uint8)

        # Resize mask to match the input image dimensions
        mask_resized = cv2.resize(mask, (W, H))

        # Save the mask as a PNG file
        output_path = f'./output_mask_{i}_{j}.png'
        cv2.imwrite(output_path, mask_resized)

        print(f"Mask {j} from result {i} saved at {output_path}")
