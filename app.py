# app.py (Render-friendly)
import os
import io
import base64
import logging
from collections import deque

from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np

# -------- Limit thread use (helps memory/CPU on free tier)
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("signlang")

# -------- App init (NO heavy imports here)
app = Flask(__name__, static_folder="static", template_folder="templates")

# Config
MODEL_PATHS = ["model/model.h5", "model.h5"]
LABELS_PATH = os.getenv("LABELS_PATH", "labels.txt")
USE_TRAINING_ROI = False   # set True if you need your old 0:300,40:400 crop

# Globals to be loaded lazily
_tf = None
_mp = None
_hands = None
_model = None
_CLASS_NAMES = None
_TIMESTEPS = 30
_FEAT_DIM = 63
BUFFER = deque(maxlen=_TIMESTEPS)

def load_labels():
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]
    # default A..Z
    import string
    return list(string.ascii_uppercase[:26])

def lazy_imports_and_model():
    """Import heavy libs & load model exactly once."""
    global _tf, _mp, _hands, _model, _CLASS_NAMES, _TIMESTEPS, _FEAT_DIM
    if _model is not None:
        return

    log.info("Loading TensorFlow (CPU) & MediaPipe lazily...")
    import tensorflow as tf
    import cv2
    import mediapipe as mp
    _tf = tf
    _mp = mp

    # Load model
    model = None
    last_err = None
    for p in MODEL_PATHS:
        if os.path.exists(p):
            try:
                model = tf.keras.models.load_model(p)
                log.info(f"Loaded model from {p}")
                break
            except Exception as e:
                last_err = e
    if model is None:
        # fallback: JSON + weights
        if os.path.exists("model.json") and os.path.exists("model.h5"):
            with open("model.json", "r") as f:
                model = tf.keras.models.model_from_json(f.read())
            model.load_weights("model.h5")
            log.info("Loaded model from model.json + model.h5")
        else:
            raise RuntimeError(f"Could not load model. Last error: {last_err}")

    # Infer expected input shape (None, T, F)
    in_shape = model.input_shape
    if isinstance(in_shape, (list, tuple)) and len(in_shape) >= 3:
        _, T, F = in_shape[:3]
        _TIMESTEPS = int(T or 30)
        _FEAT_DIM = int(F or 63)
    log.info(f"Model expects (T={_TIMESTEPS}, F={_FEAT_DIM})")

    # Load labels
    _CLASS_NAMES = load_labels()
    log.info(f"Loaded {len(_CLASS_NAMES)} labels: {_CLASS_NAMES[:5]}...")

    # Init MediaPipe Hands
    _hands = mp.solutions.hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Stash globals
    globals().update({
        "_model": model,
        "_CLASS_NAMES": _CLASS_NAMES,
        "_TIMESTEPS": _TIMESTEPS,
        "_FEAT_DIM": _FEAT_DIM,
        "_hands": _hands
    })

def decode_data_url_to_bgr(data_url: str):
    header, b64 = data_url.split(",", 1)
    raw = base64.b64decode(b64)
    pil = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(pil)  # RGB
    # lazy import cv2 only when needed
    import cv2
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def mediapipe_detection_bgr(bgr_img):
    import cv2
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = _hands.process(rgb)
    rgb.flags.writeable = True
    out_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return out_bgr, results

def extract_keypoints(results):
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        rh = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark], dtype=np.float32).flatten()
        return rh  # (63,)
    return np.zeros((63,), dtype=np.float32)

@app.route("/", methods=["GET"])
def home():
    # Light render — no heavy work here
    names = _CLASS_NAMES if _CLASS_NAMES else load_labels()
    return render_template("index.html", class_names=names)

@app.route("/health", methods=["GET"])
def health():
    # Must be fast and not trigger model load
    return jsonify({"status": "ok", "labels": len(_CLASS_NAMES or [])})

@app.route("/reset", methods=["POST"])
def reset():
    BUFFER.clear()
    return jsonify({"status": "cleared", "buffer": 0})

@app.route("/predict", methods=["POST"])
def predict():
    # Heavy stuff only kicks in on first call
    lazy_imports_and_model()

    data = request.get_json(silent=True) or {}
    data_url = data.get("image")
    if not data_url or not data_url.startswith("data:"):
        return jsonify({"error": "Send JSON: {'image': dataURL}"}), 400

    bgr = decode_data_url_to_bgr(data_url)

    if USE_TRAINING_ROI:
        H, W = bgr.shape[:2]
        y0, y1 = 40, min(400, H)
        x0, x1 = 0, min(300, W)
        bgr = bgr[y0:y1, x0:x1]

    _, results = mediapipe_detection_bgr(bgr)
    kp = extract_keypoints(results)

    # pad/trim to model feature dim
    if kp.shape[0] < _FEAT_DIM:
        kp = np.pad(kp, (0, _FEAT_DIM - kp.shape[0]), mode="constant")
    elif kp.shape[0] > _FEAT_DIM:
        kp = kp[:_FEAT_DIM]

    BUFFER.append(kp)

    if len(BUFFER) < _TIMESTEPS:
        return jsonify({"status": "collecting", "have": len(BUFFER), "need": _TIMESTEPS})

    win = np.stack(BUFFER, axis=0).astype(np.float32)  # (T,F)
    x = np.expand_dims(win, axis=0)                    # (1,T,F)

    probs = _model.predict(x, verbose=0)[0].astype(float)
    top = int(np.argmax(probs))
    label = _CLASS_NAMES[top] if top < len(_CLASS_NAMES) else str(top)
    return jsonify({
        "status": "ok",
        "label": label,
        "confidence": float(probs[top]),
        "probs": { _CLASS_NAMES[i]: float(probs[i]) for i in range(min(len(_CLASS_NAMES), len(probs))) },
        "window": _TIMESTEPS
    })

# Optional: quick zero-input test — does not load model
@app.route("/ping", methods=["GET"])
def ping():
    return "pong"
