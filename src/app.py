import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Title
st.title("🌱 Soil Suitability for Vegetable Cultivation")

st.write("Upload a soil image to detect soil type and get suitable vegetable recommendations.")

# Load model
model = tf.keras.models.load_model("soil_model_densenet.h5")

# Class labels (must match dataset folders)
soil_classes = ['Alluvial soil', 'Black soil', 'Clay soil', 'Red soil']

# Upload image
uploaded_file = st.file_uploader("📤 Upload Soil Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    # Show image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize((224,224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    confidence = np.max(prediction)
    predicted_class = soil_classes[np.argmax(prediction)]

    # 🔴 Validation (Non-soil detection)
    if confidence < 0.6:
        st.error("❌ Invalid image. Please upload a proper soil image.")
    else:
        st.success(f"✅ Predicted Soil Type: {predicted_class}")
        st.write(f"🔍 Confidence: {confidence*100:.2f}%")

        # Vegetable recommendation
        if predicted_class == "Alluvial soil":
            vegetables = ["Potato", "Tomato", "Carrot", "Spinach", "Onion"]

        elif predicted_class == "Black soil":
            vegetables = ["Onion", "Chili", "Cotton", "Brinjal (Eggplant)", "Okra (Ladyfinger)"]

        elif predicted_class == "Clay soil":
            vegetables = ["Cabbage", "Broccoli", "Peas", "Cauliflower", "Lettuce"]

        else:
            vegetables = ["Groundnut", "Millets", "Pulses (Dal)", "Sweet Potato", "Beans"]

        st.subheader("🌾 Recommended Vegetables")
        st.write(vegetables)
