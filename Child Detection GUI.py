import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from keras.models import load_model

# Load the trained model
model = load_model('child_classifier_model.h5')

# Function to preprocess the selected image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64))  # Resize image to match model input size
    image = image.astype('float32') / 255.0  # Normalize pixel values
    return image.reshape(1, 64, 64, 3)  # Reshape image for model input

# Function to make prediction when image is selected
def predict_image():
    # Open file dialog to select image
    file_path = filedialog.askopenfilename()
    if file_path:
        # Preprocess the selected image
        processed_image = preprocess_image(file_path)
        # Make prediction using the model
        prediction = model.predict(processed_image)[0][0]
        # Determine prediction label
        prediction_label = "NOT Child" if prediction >= 0.5 else "Child"
        # Display prediction
        result_label.config(text=f"Prediction: {prediction_label}")

# Create the main application window with larger size
root = tk.Tk()
root.title("Child Classifier")
root.geometry("600x400")  # Set the window size

# Create a button to select image
select_button = tk.Button(root, text="Select Image", command=predict_image)
select_button.pack(pady=20)

# Label to display prediction result
result_label = tk.Label(root, text="")
result_label.pack()

# Run the Tkinter event loop
root.mainloop()
