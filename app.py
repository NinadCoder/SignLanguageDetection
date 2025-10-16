# app.py — Flask server that:
# - accepts frames from the browser
# - extracts MediaPipe hand keypoints (21*3 = 63)
# - buffers the last 30 frames
# - runs your LSTM model and returns probs

import os, io, base64, logging, json
from collections import deque

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template

# --- DL / CV deps
import cv2
import tensorflow as tf
import mediapipe as mp

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("signlang")

# ---------------------- Config ----------------------
# Try both locations so it works with your training outputs
MODEL_CANDIDATES = ["model/model.h5", "model.h5"]
MODEL_JSON = "model.json"  # optional (if you only saved weights)
LABELS_TXT = "labels.txt"  # optional; if missing we fall back to A..Z

# If you want to mirror your old ROI (x: 0..300, y: 40..400), set this True
USE_TRAINING_ROI = False

# ---------------------- Load model ----------------------
def load_any_model():
    # 1) Try full SavedModel/H5
    for p in MODEL_CANDIDATES:
        if os.path.exists(p):
            try:
                m = tf.keras.models.load_model(p)
                log.info(f"Loaded full model from {p}")
                return m
            except Exception as e:
                log.warning(f"Failed load_model({p}): {e}")

    # 2) Fallback: model.json + weights.h5
    if os.path.exists(MODEL_JSON) and os.path.exists("model.h5"):
        with open(MODEL_JSON, "r") as f:
            m = tf.keras.models.model_from_json(f.read())
        m.load_weights("model.h5")
        log.info("Loaded model from model.json + model.h5 (weights)")
        return m

    raise RuntimeError(
        "No model found. Put your model at model/model.h5 or model.h5 "
        "or provide model.json + model.h5 (weights)."
    )

model = load_any_model()

# infer (timesteps, feat_dim) from the model itself
# expected: (None, 30, 63)
in_shape = model.input_shape
_, TIMESTEPS, FEAT_DIM = in_shape if isinstance(in_shape, (list, tuple)) else (None, 30, 63)
TIMESTEPS = TIMESTEPS or 30
FEAT_DIM = FEAT_DIM or 63
log.info(f"Model expects sequences of shape: (T={TIMESTEPS}, F={FEAT_DIM})")

# ---------------------- Labels ----------------------
def load_labels():
    if os.path.exists(LABELS_TXT):
        with open(LABELS_TXT, "r", encoding="utf-8") as f:
            labs = [ln.strip() for ln in f if ln.strip()]
        return labs
    # fallback: A..Z (your training uses actions = ['A', ... 'Z'])
    import string
    return list(string.ascii_uppercase[:26])

CLASS_NAMES = load_labels()
NUM_CLASSES = len(CLASS_NAMES)
log.info(f"Loaded {NUM_CLASSES} labels.")

# ---------------------- MediaPipe ----------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def mediapipe_detection_bgr(bgr_image):
    """Takes a BGR image, returns (same_bgr, results)."""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True
    out_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return out_bgr, results

def extract_keypoints(results):
    """Return 63-dim vector (x,y,z for 21 landmarks) or zeros if none."""
    if results.multi_hand_landmarks:
        # Use first detected hand (same as your training loop)
        hand = results.multi_hand_landmarks[0]
        rh = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32).flatten()
        return rh  # (63,)
    return np.zeros((63,), dtype=np.float32)

# ---------------------- Buffer (single-user demo) ----------------------
BUFFER = deque(maxlen=TIMESTEPS)

# ---------------------- Flask ----------------------
app = Flask(__name__, static_folder="static", template_folder="templates")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", class_names=CLASS_NAMES)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "timesteps": TIMESTEPS,
        "feat_dim": FEAT_DIM,
        "labels": CLASS_NAMES
    })

@app.route("/reset", methods=["POST"])
def reset():
    BUFFER.clear()
    return jsonify({"status": "cleared", "buffer": 0})

def decode_data_url_to_bgr(data_url: str):
    """data:image/...;base64,XXXX -> BGR numpy image"""
    header, b64data = data_url.split(",", 1)
    raw = base64.b64decode(b64data)
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(pil)  # RGB
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON:
      {"image": "<dataURL>"}  --> one frame
    Streams: call this repeatedly from the browser; server maintains a 30-frame buffer.
    """
    try:
        data = request.get_json(silent=True) or {}
        data_url = data.get("image")
        if not data_url or not data_url.startswith("data:"):
            return jsonify({"error": "Send JSON: {'image': dataURL}"}), 400

        bgr = decode_data_url_to_bgr(data_url)

        # Optionally mirror your old ROI (0:300, 40:400) if you stream larger frames.
        if USE_TRAINING_ROI:
            H, W = bgr.shape[:2]
            y0, y1 = 40, min(400, H)
            x0, x1 = 0, min(300, W)
            bgr = bgr[y0:y1, x0:x1]

        _, results = mediapipe_detection_bgr(bgr)
        kp = extract_keypoints(results)

        # If your model expects FEAT_DIM != 63 (because of padding in training), pad here:
        if kp.shape[0] < FEAT_DIM:
            kp = np.pad(kp, (0, FEAT_DIM - kp.shape[0]), mode="constant")
        elif kp.shape[0] > FEAT_DIM:
            kp = kp[:FEAT_DIM]

        BUFFER.append(kp)

        if len(BUFFER) < TIMESTEPS:
            return jsonify({
                "status": "collecting",
                "have": len(BUFFER),
                "need": TIMESTEPS
            })

        # Build (1, T, F) batch
        win = np.stack(BUFFER, axis=0).astype(np.float32)   # (T, F)
        x = np.expand_dims(win, axis=0)                     # (1, T, F)

        probs = model.predict(x, verbose=0)[0]
        probs = probs.astype(float)
        top = int(np.argmax(probs))
        return jsonify({
            "status": "ok",
            "label": CLASS_NAMES[top] if top < NUM_CLASSES else str(top),
            "confidence": float(probs[top]),
            "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(min(NUM_CLASSES, len(probs)))},
            "window": TIMESTEPS
        })

    except Exception as e:
        log.exception("Predict failed")
        return jsonify({"error": str(e)}), 500

# quick sanity check: runs model on zeros
@app.route("/predict-test", methods=["GET"])
def predict_test():
    x = np.zeros((1, TIMESTEPS, FEAT_DIM), dtype=np.float32)
    p = model.predict(x, verbose=0)[0]
    i = int(np.argmax(p))
    return jsonify({"label": CLASS_NAMES[i] if i < NUM_CLASSES else str(i), "confidence": float(p[i])})

if __name__ == "__main__":
    # use gunicorn in prod
    app.run(host="0.0.0.0", port=8000, debug=False)
