# Deepfake Detection – Full Web Project (Flask + Animated UI + Webcam)

This is a **production-ready scaffold** for a Deepfake Detection demo website with:
- 🎥 **Real-time webcam capture**
- 📤 **Image upload (drag & drop)**
- ⚡ **Loading spinner & progress states**
- 🧠 **Keras model loading** (put your `deepfake_detector_keras.h5` under `models/`)
- 🧪 **Mock Mode**: If the model file is missing, the app uses a heuristic to simulate predictions so you can demo the UI end-to-end
- 📁 **Dataset folder** under `data/dataset/` to place your images
- 🛠️ **Training script** (`scripts/train_keras.py`) to fine-tune or train your CNN
- 🧰 **Clean project structure** and `requirements.txt`

## Quick Start

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
pip install -r requirements.txt

# Optional: place your trained model
# models/deepfake_detector_keras.h5

python app.py
```

Open http://127.0.0.1:5000/

## Dataset
Put images under `data/dataset/real/` and `data/dataset/fake/`. Then run `scripts/train_keras.py` to create the model file.

## Endpoints
- `GET /` – UI
- `POST /predict` – multipart file upload
- `POST /predict_base64` – webcam snapshot (base64)
- `GET /health` – status + model loaded
- `GET /about` – metadata

## TO ACCESS DATASET -->https://drive.google.com/file/d/1QkeiNUkT6JDE97vxQy4dL_Ecdu4nqczO/view?usp=sharing
- Uses 128×128 preprocessing by default.
- Mock mode uses simple blur/edge heuristics only for demo — not for real detection.
