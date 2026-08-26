# Handwritten Digit Recognizer — Streamlit Deployment

## Files in this package
- `app.py` — the Streamlit app (upload image or use camera → predict digit)
- `requirements.txt` — pinned, tested-together dependency versions
- You must add: `handwritten_digit_cnn.keras` (your trained model)

## Step 1 — Export the model from your notebook
Your notebook already does this correctly in the second-to-last cell:

```python
model_aug.save("/content/handwritten_digit_cnn.keras")

from google.colab import files
files.download("/content/handwritten_digit_cnn.keras")
```

Run those two cells in Colab (after training finishes), and the `.keras` file
will download to your computer. **Do not skip this — `app.py` will not run
without it.**

## Step 2 — Assemble the folder
Put these 3 files in the same folder:
```
my-digit-app/
├── app.py
├── requirements.txt
└── handwritten_digit_cnn.keras
```

## Step 3 — Test locally (recommended before deploying)
```bash
pip install -r requirements.txt
streamlit run app.py
```
Open the local URL Streamlit prints, upload a digit photo, and confirm you
get a prediction.

## Step 4 — Deploy on Streamlit Community Cloud
1. Push the 3 files to a GitHub repo (the `.keras` file included — if it's
   over 100 MB, use Git LFS; a small CNN like this is usually only a few MB
   so plain git is fine).
2. Go to https://share.streamlit.io → "New app" → pick the repo/branch →
   main file path `app.py` → Deploy.
3. Streamlit Cloud installs `requirements.txt` automatically and starts the app.

## Common errors this package avoids
- **`ModuleNotFoundError: cv2`** → fixed by using `opencv-python-headless`
  instead of `opencv-python` (the GUI version fails on headless cloud servers).
- **TensorFlow/NumPy version clashes** → versions in `requirements.txt` are
  pinned to a combination known to work together.
- **`FileNotFoundError: handwritten_digit_cnn.keras`** → the model file must
  sit next to `app.py` in the same folder/repo — it is not included here
  since only you have the trained weights.
- **Wrong prediction on real-world photos** → the app uses the *exact* same
  preprocessing (grayscale → Otsu threshold → crop → square pad → resize
  32×32 → normalize) as your training notebook, so inference matches training.

## Notes on the model
- Input shape expected: `(32, 32, 1)`, values normalized to `[0, 1]`.
- The saved model includes the `RandomRotation/RandomZoom/RandomTranslation`
  augmentation layers — these are automatically inert during `model.predict()`,
  so no extra code is needed to disable them at inference time.
