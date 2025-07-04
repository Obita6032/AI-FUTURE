# Install TensorFlow if not already installed
# !pip install tensorflow scikit-learn opencv-python

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

# Simulate dataset folders
os.makedirs("dataset/plastic", exist_ok=True)
os.makedirs("dataset/metal", exist_ok=True)

# Create 10 synthetic sample images for each class
for i in range(10):
    plastic_img = np.full((128, 128, 3), (255, 255, 0), dtype=np.uint8)  # Yellow plastic
    metal_img = np.full((128, 128, 3), (192, 192, 192), dtype=np.uint8)  # Silver metal
    cv2.imwrite(f"dataset/plastic/plastic_{i}.jpg", plastic_img)
    cv2.imwrite(f"dataset/metal/metal_{i}.jpg", metal_img)

# Load and preprocess images
data, labels = [], []
for label in ["plastic", "metal"]:
    folder = f"dataset/{label}"
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = load_img(path, target_size=(128, 128))
        img = img_to_array(img)
        data.append(img)
        labels.append(label)

data = np.array(data, dtype="float32") / 255.0
labels = LabelBinarizer().fit_transform(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Build model
base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(128, 128, 3)))
base_model.trainable = False

x = base_model.output
x = AveragePooling2D(pool_size=(4, 4))(x)
x = Flatten(name="flatten")(x)
x = Dense(64, activation="relu")(x)
x = Dropout(0.3)(x)
x = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=x)
model.compile(loss="binary_crossentropy", optimizer=Adam(1e-4), metrics=["accuracy"])

# Train (quick training due to small data)
model.fit(X_train, y_train, epochs=3, batch_size=4, validation_data=(X_test, y_test))

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save model
with open("recyclable_classifier.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model converted and saved as recyclable_classifier.tflite")
