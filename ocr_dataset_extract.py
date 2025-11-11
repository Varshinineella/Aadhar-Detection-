import os
import cv2
import pytesseract
import pandas as pd
import re

# ✅ Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✅ Folder with Aadhaar images
folder_path = r"C:\Users\hp\Desktop\infosys\sample_docs"

# ✅ Create a list to store extracted data
results = []

def preprocess_image(img):
    """Enhance image quality for better OCR accuracy."""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Remove noise
    gray = cv2.medianBlur(gray, 3)
    # Thresholding for better contrast
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Resize for better OCR detection
    resized = cv2.resize(thresh, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    return resized

# ✅ Loop through all images in the folder
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(folder_path, filename)
        print(f"🔍 Processing: {filename}")

        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not read {filename}")
            continue

        # Preprocess image
        processed_img = preprocess_image(img)

        # Extract text using Tesseract
        text = pytesseract.image_to_string(processed_img, lang='eng')

        # ✅ Extract key Aadhaar details using regex
        name_match = re.search(r'Name[:\-]?\s*([A-Za-z\s]+)', text)
        dob_match = re.search(r'DOB[:\-]?\s*(\d{2}/\d{2}/\d{4})', text)
        aadhaar_match = re.search(r'\b\d{4}\s\d{4}\s\d{4}\b', text)

        name = name_match.group(1).strip() if name_match else "Not Found"
        dob = dob_match.group(1).strip() if dob_match else "Not Found"
        aadhaar = aadhaar_match.group(0).strip() if aadhaar_match else "Not Found"

        # Store results
        results.append({
            'Filename': filename,
            'Name': name,
            'DOB': dob,
            'Aadhaar Number': aadhaar
        })

# ✅ Convert results to CSV
df = pd.DataFrame(results)
output_csv = os.path.join(r"C:\Users\hp\Desktop\infosys", "aadhaar_extracted_preprocessed.csv")
df.to_csv(output_csv, index=False)

print("\n✅ OCR Extraction (with preprocessing) completed successfully!")
print(f"Results saved in: {output_csv}")

