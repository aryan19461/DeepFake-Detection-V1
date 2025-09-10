import os
import gradio as gr
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import json

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "deepfake_detector_keras.h5")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels.json")

# Load model and labels
model = load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

IMG_SIZE = (128, 128)

def predict(img):
    img = np.array(img)
    img = cv2.resize(img, IMG_SIZE) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]
    label = "Real" if pred > 0.5 else "Fake"
    confidence = pred if label == "Real" else 1 - pred

    return {
        "Real": float(confidence) if label == "Real" else float(1 - confidence),
        "Fake": float(confidence) if label == "Fake" else float(1 - confidence)
    }

# Gradio interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Deepfake Detector (Minor Project)",
    description="Upload an image or use webcam to classify as Real or Fake."
)

if __name__ == "__main__":
    demo.launch()
