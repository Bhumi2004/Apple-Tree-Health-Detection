import os
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt

# Load the model
loaded_model = tf.keras.models.load_model("leaf_classification_model.keras")

# Corrected image path
img_path = r"C:\Users\BHUMI\Desktop\Apple Leaf Detection\apple_healthy5.jpg"  # Make sure the path is correct

# Verify if the image file exists before loading
if os.path.exists(img_path):
    # Load an image to test
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to match the input size of the model

    # Convert image to array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image

    # Predict using the loaded model
    prediction = loaded_model.predict(img_array)

    # Interpret the prediction
    print(f"Model output: {prediction[0]}")  # Debug: print model output
    if prediction[0] < 0.5:
        print("Prediction: Healthy Apple")
    else:
        print("Prediction: Scab-Affected Apple")

    # Display the image
    plt.imshow(img)
    plt.title(f"Model output: {prediction[0]}")
    plt.axis('off')
    plt.show()

else:
    print(f"File not found: {img_path}. Please check the file path and try again.")

    print(f"File not found: {img_path}. Please check the file path and try again.")

