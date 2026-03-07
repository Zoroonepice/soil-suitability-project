import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load trained model
model = tf.keras.models.load_model("soil_model.h5")

# Soil classes
soil_classes = ['alluvial', 'black', 'clay', 'red']

# Load soil image
img = image.load_img("test_soil.jpg", target_size=(224,224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict soil type
prediction = model.predict(img_array)
predicted_class = soil_classes[np.argmax(prediction)]

print("Predicted Soil Type:", predicted_class)

if predicted_class == "alluvial":
    vegetables = ["Potato", "Tomato", "Carrot"]

elif predicted_class == "black":
    vegetables = ["Onion", "Chili", "Cotton"]

elif predicted_class == "clay":
    vegetables = ["Cabbage", "Broccoli", "Peas"]

elif predicted_class == "red":
    vegetables = ["Groundnut", "Millets", "Pulses"]

print("Recommended Vegetables:", vegetables)