from flask import Flask, request, jsonify
import cv2
import numpy as np
import base64
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

def convert_to_sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blur = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    return sketch

@app.route("/sketch", methods=["POST"])
def sketch():
    file = request.files["image"]
    if not file:
        return jsonify({"error": "No image uploaded"}), 400

    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    sketch_image = convert_to_sketch(image)
    _, buffer = cv2.imencode(".png", sketch_image)
    sketch_base64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({"sketch": sketch_base64})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port, debug=True)
