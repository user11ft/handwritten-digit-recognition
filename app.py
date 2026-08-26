"""
Handwritten Digit Recognition — Streamlit App
------------------------------------------------
Loads the trained CNN (handwritten_digit_cnn.keras) and predicts a digit (0-9)
from an uploaded image or a webcam snapshot.

Preprocessing pipeline mirrors the training notebook exactly:
Raw image -> Grayscale -> Otsu threshold -> largest-contour crop -> square pad
-> resize 32x32 -> normalize [0,1] -> add channel dim.

Place this file, `requirements.txt`, and `handwritten_digit_cnn.keras`
in the same folder before deploying.
"""

import io

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    page_icon="✍️",
    layout="centered",
)

MODEL_PATH = "handwritten_digit_cnn.keras"
IMG_SIZE = 32


# ----------------------------------------------------------------------
# Model loading (cached so it only loads once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# ----------------------------------------------------------------------
# Preprocessing — identical logic to the training notebook,
# adapted to work on an in-memory numpy array instead of a file path.
# ----------------------------------------------------------------------
def preprocess_image(pil_image: Image.Image):
    # 1. PIL -> OpenCV BGR array
    rgb = np.array(pil_image.convert("RGB"))
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # 2. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 3. Otsu threshold (inverted: digit becomes white on black)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # 4. Find contours
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        return None, None

    # 5. Largest contour = the digit
    contour = max(contours, key=cv2.contourArea)

    # 6. Bounding box
    x, y, w, h = cv2.boundingRect(contour)

    # 7. Add margin
    margin = 10
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(binary.shape[1], x + w + margin)
    y2 = min(binary.shape[0], y + h + margin)

    cropped = binary[y1:y2, x1:x2]
    if cropped.size == 0:
        return None, None

    # 8. Pad to square
    h, w = cropped.shape
    size = max(h, w)
    square = np.zeros((size, size), dtype=np.uint8)
    x_offset = (size - w) // 2
    y_offset = (size - h) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = cropped

    # 9. Resize to 32x32
    resized = cv2.resize(square, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    # 10. Normalize
    normalized = resized.astype("float32") / 255.0

    # 11. Add channel dimension -> (32, 32, 1)
    normalized = np.expand_dims(normalized, axis=-1)

    return normalized, resized  # model input, and a viewable 8-bit copy


def predict_digit(model, processed):
    batch = np.expand_dims(processed, axis=0)  # (1, 32, 32, 1)
    probs = model.predict(batch, verbose=0)[0]
    digit = int(np.argmax(probs))
    confidence = float(np.max(probs))
    return digit, confidence, probs


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("✍️ Handwritten Digit Recognizer")
st.caption("Upload a photo of a handwritten digit (0–9) or take one with your camera.")

try:
    model = load_model()
except Exception as e:
    st.error(
        f"Could not load `{MODEL_PATH}`. Make sure the .keras file is in the "
        f"same folder as this app.\n\nDetails: {e}"
    )
    st.stop()

tab_upload, tab_camera = st.tabs(["📁 Upload / Paste Image", "📷 Use Camera"])

image_file = None

with tab_upload:
    st.write("Drag & drop, browse, or **paste** (Ctrl/Cmd+V) an image below.")
    image_file = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg", "bmp"],
        accept_multiple_files=False,
        key="uploader",
    )

with tab_camera:
    camera_file = st.camera_input("Take a picture of a handwritten digit")
    if camera_file is not None:
        image_file = camera_file

# ----------------------------------------------------------------------
# Run prediction
# ----------------------------------------------------------------------
if image_file is not None:
    pil_image = Image.open(io.BytesIO(image_file.getvalue()))

    processed, preview = preprocess_image(pil_image)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_image, caption="Original", use_container_width=True)

    if processed is None:
        st.warning(
            "⚠️ No digit could be detected in this image. Try a clearer photo "
            "with good contrast between the digit and the background."
        )
    else:
        with col2:
            st.image(preview, caption="Processed (32×32)", clamp=True, use_container_width=True)

        if st.button("🔮 Predict Digit", type="primary", use_container_width=True):
            digit, confidence, probs = predict_digit(model, processed)

            st.success(f"### Predicted digit: **{digit}**")
            st.write(f"Confidence: **{confidence:.2%}**")

            probs_df = pd.DataFrame(
                {"Digit": [str(i) for i in range(10)], "Probability": probs}
            ).set_index("Digit")
            st.bar_chart(probs_df)
else:
    st.info("Upload or capture an image to get a prediction.")
