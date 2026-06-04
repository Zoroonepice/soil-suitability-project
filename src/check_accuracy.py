import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model
model = tf.keras.models.load_model("soil_model_densenet.h5")

# Dataset path
test_path = "Dataset/test"

# Preprocess test data
test_datagen = ImageDataGenerator(rescale=1./255)

test_data = test_datagen.flow_from_directory(
    test_path,
    target_size=(224,224),
    batch_size=8,
    class_mode='categorical'
)

# Evaluate model
loss, accuracy = model.evaluate(test_data)

print("Test Accuracy:", accuracy)
