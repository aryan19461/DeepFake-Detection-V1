import os
import io
import base64
import uuid
import numpy as np
from flask import Flask, render_template, request, jsonify
from PIL import Image, ImageOps, ImageFilter

# Optional keras import (will be used if model is present)
MODEL = None
MODEL_INPUT_SIZE = (128, 128)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "models", "deepfake_detector_keras.h5")

os.makedirs(UPLOAD_DIR, exist_ok=True)

# Try to load Keras model if present
def try_load_model():
    global MODEL
    if os.path.exists(MODEL_PATH):
        try:
            from tensorflow.keras.models import load_model
            MODEL = load_model(MODEL_PATH)
            print("[INFO] Keras model loaded:", MODEL_PATH)
        except Exception as e:
            print("[WARN] Failed to load model:", e)
            MODEL = None
    else:
        MODEL = None

try_load_model()

def pil_to_array(img: Image.Image):
    # Convert PIL image to normalized numpy array for model
    img = img.convert("RGB").resize(MODEL_INPUT_SIZE)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def mock_fake_probability(img: Image.Image) -> float:
    try:
        import cv2
        img_cv = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        var_lap = cv2.Laplacian(img_cv, cv2.CV_64F).var()
    except Exception:
        var_lap = 50.0

    # Edge density proxy
    edges = ImageOps.autocontrast(img.convert("L")).filter(ImageFilter.FIND_EDGES)
    edge_mean = np.mean(np.array(edges))

    # Normalize proxies and map to 0..1
    sharp = np.tanh(var_lap / 200.0)  # higher var_lap -> sharper
    edginess = np.tanh(edge_mean / 64.0)

    # Combine proxies into a pseudo fake probability
    score = 0.6 * (1 - sharp) + 0.4 * (1 - edginess)
    score = float(np.clip(score, 0, 1))
    return score

def predict_pil(img: Image.Image):
    """
    Returns a dict with {prob_fake: float, label: str, model_loaded: bool}
    """
    if MODEL is not None:
        arr = pil_to_array(img)
        try:
            prob = float(MODEL.predict(arr)[0][0])  # Assuming sigmoid output
        except Exception as e:
            print("[WARN] Model predict failed, fallback to mock:", e)
            prob = mock_fake_probability(img)
        model_loaded = True
    else:
        prob = mock_fake_probability(img)
        model_loaded = False

    label = "FAKE" if prob >= 0.5 else "REAL"
    return {"prob_fake": prob, "label": label, "model_loaded": model_loaded}

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": MODEL is not None})

@app.route("/about")
def about():
    return jsonify({
        "project": "Deepfake Detection Web App",
        "framework": "Flask",
        "model_expected": os.path.basename(MODEL_PATH),
        "model_loaded": MODEL is not None,
        "input_size": MODEL_INPUT_SIZE
    })

@app.route("/predict", methods=["POST"])
def predict_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Save then open with PIL
    file_id = str(uuid.uuid4())[:8]
    save_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    file.save(save_path)

    try:
        img = Image.open(save_path)
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    result = predict_pil(img)
    result["filename"] = os.path.basename(save_path)
    return jsonify(result)

@app.route("/predict_base64", methods=["POST"])
def predict_base64():
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "Missing 'image' base64 field"}), 400

    b64 = data["image"]
    if "," in b64:
        b64 = b64.split(",", 1)[1]  # handle data URL

    try:
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw))
    except Exception as e:
        return jsonify({"error": f"Invalid base64 image: {e}"}), 400

    result = predict_pil(img)
    return jsonify(result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
