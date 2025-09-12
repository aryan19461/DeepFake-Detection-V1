# app.py — robust label/order handling + correct input size
import json, cv2, numpy as np, gradio as gr
from tensorflow import keras
from explain_utils import grad_cam, artifact_reasons

MODEL_PATH = "models/deepfake_detector_keras.h5"
LABELS_JSON = "models/labels.json"  # keep it next to the model
IMG_SIZE = (128, 128)               # <-- minor model input size

# Load model
model = keras.models.load_model(MODEL_PATH, compile=False)

# Load labels (supports {"0":"Real","1":"Fake"} or {"Fake":0,"Real":1})
try:
    with open(LABELS_JSON, "r") as f:
        raw = json.load(f)
except Exception:
    raw = {"0": "Real", "1": "Fake"}

# Build idx2label and label2idx regardless of format
if all(str(k).isdigit() for k in raw.keys()):           # index->name (str keys)
    idx2label = {int(k): str(v) for k, v in raw.items()}
    label2idx = {v: k for k, v in idx2label.items()}
elif all(isinstance(v, int) for v in raw.values()):      # name->index
    label2idx = {str(k): int(v) for k, v in raw.items()}
    idx2label = {v: k for k, v in label2idx.items()}
else:                                                     # fallback
    idx2label = {0: "Fake", 1: "Real"}
    label2idx = {"Fake": 0, "Real": 1}

# Helper: map a prediction vector to probs aligned to indices [0,1]
def _postprocess_probs(pred_vec):
    pred = np.array(pred_vec).reshape(-1)
    n_classes = max(idx2label.keys()) + 1 if idx2label else (pred.size or 2)

    if pred.size == 1 and n_classes == 2:
        # Sigmoid => p1 = P(class index 1); p0 = 1 - p1
        p1 = float(pred[0])
        probs = np.zeros(2, dtype=np.float32)
        probs[1] = p1
        probs[0] = 1.0 - p1
    elif pred.size == n_classes:
        probs = pred.astype(np.float32)
    else:
        # Fallback normalization
        probs = pred.astype(np.float32)
        if probs.size: probs = probs / (probs.sum() + 1e-8)
        probs = np.pad(probs, (0, max(0, 2 - probs.size)))[:2]
    pred_idx = int(np.argmax(probs))
    pred_conf = float(probs[pred_idx])
    return probs, pred_idx, pred_conf

def predict_and_explain(img):
    if img is None:
        return {"error": "No image provided"}, None, "Please upload or capture an image."

    img_rgb = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    x = cv2.resize(img_rgb, IMG_SIZE).astype("float32") / 255.0
    x = np.expand_dims(x, 0)

    raw = model.predict(x, verbose=0)[0]
    probs, pred_idx, conf = _postprocess_probs(raw)
    pred_label = idx2label.get(pred_idx, f"class{pred_idx}")

    try:
        heatmap, overlay_bgr = grad_cam(model, bgr, target_size=IMG_SIZE, target_class=pred_idx)
    except Exception:
        overlay_bgr = bgr.copy()
        heatmap = np.zeros((bgr.shape[0], bgr.shape[1]), dtype=np.float32)

    try:
        reasons = artifact_reasons(bgr, heatmap)
    except Exception as e:
        reasons = [f"Explainability unavailable: {e}"]
    reasons_txt = "\n".join([f"• {r}" for r in reasons])

    overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
    return {"Prediction": pred_label, "Confidence": round(conf, 4)}, overlay_rgb, reasons_txt

with gr.Blocks(title="Deepfake Detector • Explainable", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Deepfake Detector\nUpload or capture an image, then press **Analyze**.")

    with gr.Row():
        in_img = gr.Image(type="pil", sources=["upload","webcam"], label="Input Image", height=300)
        analyze_btn = gr.Button("Analyze", variant="primary")

    with gr.Row():
        out_json = gr.JSON(label="Prediction & Confidence")
        out_overlay = gr.Image(type="numpy", label="Grad-CAM Attention", height=300)
    out_reasons = gr.Textbox(label="Why this looks fake (heuristics)", lines=5)

    analyze_btn.click(predict_and_explain, inputs=[in_img], outputs=[out_json, out_overlay, out_reasons])

if __name__ == "__main__":
    demo.launch()
