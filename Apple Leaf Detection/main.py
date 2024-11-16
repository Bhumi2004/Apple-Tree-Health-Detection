import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import os

# Define paths
data_dir = r"C:\Users\BHUMI\Desktop\new\leafdetect" 

# Check if dataset directory exists
if not os.path.exists(data_dir):
    raise ValueError(f"Dataset directory '{data_dir}' does not exist.")

# Image data generator with augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255,           # Normalize pixel values
    rotation_range=20,         # Rotate images
    width_shift_range=0.2,     # Shift images horizontally
    height_shift_range=0.2,    # Shift images vertically
    shear_range=0.2,           # Shear transformation
    zoom_range=0.2,            # Zoom in/out
    horizontal_flip=True,      # Flip images horizontally
    fill_mode='nearest'
)

# Load data from the directory
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),    # Resize images to fit model input
    batch_size=2,              # Small batch size due to small dataset
    class_mode='binary'        # Binary classification: healthy neem vs guava
)

# Load pre-trained MobileNetV2 and exclude the top classification layer
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the base model to avoid changing pre-trained weights
base_model.trainable = False

# Build the model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),  # Pooling layer to reduce dimensions
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_data,
    epochs=10,             # Use a small number of epochs due to limited data
    steps_per_epoch=len(train_data)  # Number of steps per epoch
)

# Evaluate the model
loss, accuracy = model.evaluate(train_data)
print(f"Loss: {loss}, Accuracy: {accuracy}")

# Save the model
model.save("leaf_classification_model.keras")
print("Model saved as 'leaf_classification_model.h5'")
