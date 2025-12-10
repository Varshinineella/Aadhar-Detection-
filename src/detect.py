import os
from ultralytics import YOLO
import cv2

# Path to your single Aadhaar image
img_path = r"C:\Users\hp\Desktop\infosys\fraud_detection_project\src\data1\aadhaar.jpg"

# Output folder
output_folder = r"C:\Users\hp\Desktop\infosys\fraud_detection_project\results"
os.makedirs(output_folder, exist_ok=True)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Run YOLO detection
results = model(img_path)

# Annotated image
annotated = results[0].plot()
if annotated is None:
    annotated = results[0].orig_img

# Save result
filename = os.path.basename(img_path)
save_path = os.path.join(output_folder, filename)
cv2.imwrite(save_path, annotated)
print("saved:", save_path)

print("DONE! Check folder:", output_folder)
