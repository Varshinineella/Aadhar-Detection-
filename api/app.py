from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])  # ✅ must include POST here
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    # Here you can process the file, e.g., save or run prediction
    return jsonify({"message": f"File '{file.filename}' received successfully"})

if __name__ == "__main__":
    app.run(debug=True)
