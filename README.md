# Sign Language Detector — Flask Demo

A clean Flask web app to demo your trained `model.h5` to teachers with a webcam UI and a probability bar chart.

## Quick start

1. Put your trained Keras model at: `model/model.h5`
2. List your class labels (one per line) in `labels.txt`
3. Create a virtualenv and install deps:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the dev server:
   ```bash
   python app.py
   # open http://localhost:8000
   ```

## Notes

- The current preprocessing assumes an image model that takes 224×224 RGB normalized to [0,1].
- If your model expects **keypoints / sequences (e.g., MediaPipe + LSTM)**, replace `preprocess_pil` with your feature extraction and sequence building code before `model.predict(...)`.
- For production, use gunicorn: `gunicorn -w 2 -b 0.0.0.0:8000 app:app`

## Files
- `app.py` — Flask server with `/predict`
- `templates/index.html` — Tailwind + Chart.js UI
- `static/app.js` — Camera, capture, upload, and chart logic
- `labels.txt` — Example labels (edit this for your model)
- `model/model.h5` — **Place your model here** (not included)
