import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ======== CONFIG ========
IMG_SIZE = 64
IS_RGB = True  # Set to False for grayscale
CLASS_NAMES = ['rose', 'sunflower', 'tulip', 'daisy', 'dandelion']

# ======== LOAD MODEL ========
model = tf.keras.models.load_model("saved_model.keras")

# ======== PREDICTION FUNCTION ========
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image)

    if not IS_RGB:
        if img_array.ndim == 3:
            img_array = img_array[:, :, 0]  # Take one channel
        img_array = np.expand_dims(img_array, axis=-1)

    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(image):
    processed = preprocess_image(image)
    logits = model.predict(processed)[0]
    prediction = np.argmax(logits)
    return CLASS_NAMES[prediction], tf.nn.softmax(logits)[prediction].numpy()

# ======== STREAMLIT UI ========
st.title("🧠 Image Classifier")
st.write("Upload an image to classify it using the trained model.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    label, confidence = predict(image)
    st.success(f"Predicted Class: **{label}**")
    st.info(f"Confidence: {confidence:.2%}")
