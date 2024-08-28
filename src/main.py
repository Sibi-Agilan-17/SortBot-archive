import logging
import os
import random
import sys
import threading
import time

import cv2
import numpy as np
import playsound
import tensorflow as tf
import tkinter as tk

from tensorflow.keras.preprocessing import image_dataset_from_directory, image
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk


data_dir = "./dataset-resized"
audio_dir = "./assets/audio"
MODEL = "trash_recognition_model.h5"


def load_model(filename: str = MODEL) -> tf.keras.models.Model:
    """Load the model from the file if it exists, otherwise return a new model."""

    try:
        model = tf.keras.models.load_model(filename)
        model.summary()
        return model

    except (FileNotFoundError, OSError) as exc:
        logging.error(f'Error loading model: {exc}')
        sys.exit(-1)


def generate_dataset(subset: str = "training", split: float = 0.01, batch_size: int = 16) -> tf.data.Dataset:
    """Generate a dataset from the directory."""

    return image_dataset_from_directory(
        data_dir,
        validation_split=split,
        subset=subset,
        seed=random.randint(2 ** 16, 2 ** 32 - 1),
        image_size=(256, 256),  # resize if needed
        batch_size=batch_size
    )


def load_image() -> None:
    """Load an image from the file system and display it in the GUI."""
    file_path = filedialog.askopenfilename()

    if file_path:
        display_image(file_path)


def display_image(image_path):
    """Display the image in the GUI."""
    img = Image.open(image_path)
    img = img.resize((256, 256), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    panel.config(image=str(img))
    panel.image = img
    panel.image_path = image_path


def predict_image() -> None:
    """Predict the class of the displayed image."""
    if not hasattr(panel, 'image_path'):
        messagebox.showerror("Error", "No image to predict.")
        return

    img_path = panel.image_path
    model = load_model()  # Load the model
    predicted_class = predict_image_from_path(img_path, model)
    messagebox.showinfo("Prediction", f"Predicted class: {predicted_class}")

    # Play the video in parallel after prediction
    video_thread = threading.Thread(target=play_video, args=("./assets/visual/visuals.mp4",))
    video_thread.start()


def predict_image_from_path(image_path: str, model: tf.keras.models.Model) -> str:
    """Predict the class of an image."""
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis

    # Normalize the image array
    img_array /= 255.0

    # Predict the class
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    score = np.array(score)
    ind = np.argmax(score)
    # return R for Renewable and N for Non-Renewable
    predicted_class = "R" if ind in [0, 3] else "N"  # 0 for cardboard 3 for paper

    print(f'Predicted class: {predicted_class}')
    return predicted_class


def play_video(video_path: str):
    """Play a video from the specified file path."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Video', frame)

        # Press 'q' to exit the video playback
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def capture_image():
    """Capture an image from the computer camera and save it to the specified path."""

    save_path = "captured_image.jpg"
    # Initialize the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Capture a single frame
    ret, frame = cap.read()

    if ret:
        # Save the captured image
        cv2.imwrite(save_path, frame)
        print(f"Image saved to {save_path}")
    else:
        print("Error: Could not capture image.")

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()


# Create the main window
root = tk.Tk()
root.title("Image Prediction GUI")

# Create and place the buttons with blue background
capture_button = tk.Button(root, text="Capture Image", command=capture_image, bg="lightblue")
capture_button.pack(side="top", fill="both", expand=1, padx=10, pady=10)

load_button = tk.Button(root, text="Load Image", command=load_image, bg="lightblue")
load_button.pack(side="top", fill="both", expand=1, padx=10, pady=10)

predict_button = tk.Button(root, text="Predict Image", command=predict_image, bg="lightblue")
predict_button.pack(side="top", fill="both", expand=1, padx=10, pady=10)

# Create and place the image panel
panel = tk.Label(root)
panel.pack(side="top", fill="both", expand=1, padx=10, pady=10)

# Start the GUI event loop
root.mainloop()
