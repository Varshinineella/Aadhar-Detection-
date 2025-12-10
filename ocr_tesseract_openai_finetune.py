import os, re, cv2, pytesseract, pandas as pd
from openai import OpenAI

# ✅ Set Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ✅ Initialize OpenAI client
client = OpenAI(api_key="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")

# ✅ Aadhaar dataset folder
image_folder = r"C:\Users\hp\Desktop\infosys\sample_docs"

# ✅ List to store final cleaned data
final_data = []

def preprocess_image(img):
    """Preprocess image for better OCR clarity"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thresh

def correct_with_openai(raw_text):
    """Ask OpenAI to clean and extract structured Aadhaar info"""
    prompt = f"""
    You are an AI assistant improving OCR text from Aadhaar cards.
    Clean and format the following text to clearly extract:
    - Name
    - Date of Birth (DD/MM/YYYY)
    - Aadhaar Number (#### #### ####)

    Text:
    {raw_text}

    Respond in JSON format:
    {{ "Name": "...", "DOB": "...", "Aadhaar Number": "..." }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You correct OCR text for Aadhaar data."},
                  {"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()

# ✅ Process all Aadhaar images
for file in os.listdir(image_folder):
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(image_folder, file)
        print(f"🔍 Processing: {file}")

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ Skipping unreadable file: {file}")
            continue

        # Step 1: Extract raw OCR text
        processed_img = preprocess_image(img)
        raw_text = pytesseract.image_to_string(processed_img, lang='eng')

        # Step 2: Fine-tune with OpenAI
        cleaned_output = (raw_text)

        # Step 3: Use regex as backup extraction
        name = re.search(r'"Name"\s*:\s*"([^"]+)"', cleaned_output)
        dob = re.search(r'"DOB"\s*:\s*"([^"]+)"', cleaned_output)
        aadhaar = re.search(r'"Aadhaar Number"\s*:\s*"([^"]+)"', cleaned_output)

        name = name.group(1) if name else "Not Found"
        dob = dob.group(1) if dob else "Not Found"
        aadhaar = aadhaar.group(1) if aadhaar else "Not Found"

        # Step 4: Add to final dataset
        final_data.append({
            "Filename": file,
            "Name": name,
            "DOB": dob,
            "Aadhaar Number": aadhaar
        })

# ✅ Save results to CSV
df = pd.DataFrame(final_data)
output_csv = os.path.join(r"C:\Users\hp\Desktop\infosys", "aadhaar_finetuned_verified.csv")
df.to_csv(output_csv, index=False)

print("\n✅ Fine-tuned OCR completed successfully!")
print(f"📄 Final verified file: {output_csv}")
