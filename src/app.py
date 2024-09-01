import sqlite3
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import playsound
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load TensorFlow model
model = tf.keras.models.load_model('trash_recognition_model.h5')

# Database setup
def setup_database():
    with sqlite3.connect('users.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL
            )
        ''')
        conn.commit()

setup_database()

def register_user():
    username = username_entry.get()
    password = password_entry.get()
    if username and password:
        try:
            with sqlite3.connect('users.db') as conn:
                cursor = conn.cursor()
                cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
                conn.commit()
            messagebox.showinfo("Success", "Registration successful")
        except sqlite3.IntegrityError:
            messagebox.showerror("Error", "Username already exists")
    else:
        messagebox.showerror("Error", "Please enter both username and password")

def login():
    username = username_entry.get()
    password = password_entry.get()
    with sqlite3.connect('users.db') as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
    if user:
        messagebox.showinfo("Login Success", "Welcome!")
        play_video_and_capture_camera()
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

def play_video_and_capture_camera():
    for widget in login_screen.winfo_children():
        widget.destroy()

    video_label = tk.Label(login_screen)
    video_label.pack(expand=True, fill="both")

    camera_label = tk.Label(login_screen)
    camera_label.pack(expand=True, fill="both")

    prediction_label = tk.Label(login_screen, text="", bg="#2E3440", fg="#D8DEE9", font=("Helvetica", 16))
    prediction_label.pack(pady=10)  # Ensure the label is packed

    video_path = "assets/visual/welcome.mp4"
    cap_video = cv2.VideoCapture(video_path)
    cap_camera = cv2.VideoCapture(0)

    def stream_video_and_camera():
        ret_video, frame_video = cap_video.read()
        ret_camera, frame_camera = cap_camera.read()

        if ret_video:
            frame_video = cv2.cvtColor(frame_video, cv2.COLOR_BGR2RGB)
            frame_video_image = ImageTk.PhotoImage(Image.fromarray(frame_video))
            video_label.config(image=frame_video_image)
            video_label.image = frame_video_image
        else:
            cap_video.release()

        if ret_camera:
            frame_camera = cv2.cvtColor(frame_camera, cv2.COLOR_BGR2RGB)
            frame_camera_image = ImageTk.PhotoImage(Image.fromarray(frame_camera))
            camera_label.config(image=frame_camera_image)
            camera_label.image = frame_camera_image

            # Predict using TensorFlow model
            frame_resized = cv2.resize(frame_camera, (256, 256))  # Resize to 256x256
            frame_expanded = np.expand_dims(frame_resized, axis=0)  # Expand dimensions to match model input

            predictions = model.predict(frame_expanded)

            score = tf.nn.softmax(predictions[0])
            score = np.array(score)
            ind = np.argmax(score)

            # find the class
            predicted_class = {0: "cardboard", 1: "glass", 2: "metal", 3: "paper", 4: "plastic", 5: "trash"}[ind]

            is_bio_degradable = True if ind in [0, 3] else False

            print(f'Predicted class: {predicted_class}')
            print(f'Is bio-degradable: {is_bio_degradable}')
            print(f'Confidence: {100 * np.max(score):.2f}%')

            if 100 * np.max(score) > 30:
                if is_bio_degradable:
                    playsound.playsound("assets/audio/green_bin_audio.mp3")
                else:
                    playsound.playsound("assets/audio/red_bin_audio.mp3")
                playsound.playsound("assets/audio/points.mp3")
                prediction_label.config(text=f'Predicted class: {predicted_class}\nIs bio-degradable: {is_bio_degradable}\nConfidence: {100 * np.max(score):.2f}%')
        else:
            cap_camera.release()

        login_screen.after(10, stream_video_and_camera)

    stream_video_and_camera()

def on_enter(event, next_widget=None):
    if next_widget:
        next_widget.focus_set()
    else:
        login()

def create_login_screen():
    global login_screen, username_entry, password_entry
    login_screen = tk.Tk()
    login_screen.title("Login")
    login_screen.geometry("640x840")  # Resize the window to accommodate the video
    login_screen.configure(bg="#2E3440")

    font_large = ("Helvetica", 16)
    font_medium = ("Helvetica", 14)

    tk.Label(login_screen, text="Username", bg="#2E3440", fg="#D8DEE9", font=font_large).pack(pady=10)
    username_entry = tk.Entry(login_screen, bg="#3B4252", fg="#D8DEE9", font=font_medium)
    username_entry.pack(pady=10)
    username_entry.bind("<Return>", lambda event: on_enter(event, password_entry))

    tk.Label(login_screen, text="Password", bg="#2E3440", fg="#D8DEE9", font=font_large).pack(pady=10)
    password_entry = tk.Entry(login_screen, show="*", bg="#3B4252", fg="#D8DEE9", font=font_medium)
    password_entry.pack(pady=10)
    password_entry.bind("<Return>", lambda event: on_enter(event))

    tk.Button(login_screen, text="Login", command=login, bg="#4C566A", fg="#ECEFF4", font=font_medium).pack(pady=10)
    tk.Button(login_screen, text="Register", command=register_user, bg="#4C566A", fg="#ECEFF4", font=font_medium).pack(pady=10)

    username_entry.focus_set()
    login_screen.mainloop()

create_login_screen()