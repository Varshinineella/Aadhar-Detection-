from ultralytics import YOLO
from flask import Flask, request, jsonify
import cv2


model = YOLO(r"C:\Users\hp\runs\detect\train5\weights\best.pt")


  # your trained model

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Flask server is running!"
# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['image']
    img_path = "temp.jpg"
    file.save(img_path)

    results = model.predict(img_path)

    return jsonify({
        "message": "fraud check completed",
        "results": results[0].tojson()
    })

if __name__ == "__main__":
    app.run(debug=True)

