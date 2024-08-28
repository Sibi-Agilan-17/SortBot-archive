import tkinter as tk
from tkinter import messagebox
import sqlite3
import hashlib

# Database setup
conn = sqlite3.connect('users.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT,
                points INTEGER DEFAULT 0)''')
conn.commit()

# Function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Function to login or create a new user
def login():
    global username
    username = username_entry.get()
    password = hash_password(password_entry.get())
    c.execute('SELECT points FROM users WHERE username=? AND password=?', (username, password))
    result = c.fetchone()
    if result:
        global points
        points = result[0]
        points_label.config(text=f"Points: {points}")
        login_frame.pack_forget()
        main_frame.pack()
        root.config(menu=menu_bar)  # Show menu bar when logged in
    else:
        messagebox.showerror("Login Failed", "Invalid username or password")

# Function to register a new user
def register():
    username = username_entry.get()
    password = hash_password(password_entry.get())
    try:
        c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
        conn.commit()
        messagebox.showinfo("Registration Successful", "User registered successfully")
    except sqlite3.IntegrityError:
        messagebox.showerror("Registration Failed", "Username already exists")

# Function to handle Enter key press
def handle_enter(event):
    if username_entry.focus_get() == username_entry:
        password_entry.focus()
    elif password_entry.focus_get() == password_entry:
        if username_entry.get() and password_entry.get():
            login()

# Function to increase points
def increase_points():
    global points
    points += 1
    points_label.config(text=f"Points: {points}")
    c.execute('UPDATE users SET points=? WHERE username=?', (points, username))
    conn.commit()

# Function to decrease points
def decrease_points():
    global points
    if points > 0:
        points -= 1
        points_label.config(text=f"Points: {points}")
        c.execute('UPDATE users SET points=? WHERE username=?', (points, username))
        conn.commit()

# Function to reset points
def reset_points():
    global points
    points = 0
    points_label.config(text=f"Points: {points}")
    c.execute('UPDATE users SET points=? WHERE username=?', (points, username))
    conn.commit()

# Function to log out
def logout():
    main_frame.pack_forget()
    login_frame.pack()
    root.config(menu="")  # Hide menu bar when logged out

# Function to quit the application
def quit_app():
    root.quit()

# Function to show about message
def show_about():
    messagebox.showinfo("About", "Sort Bot Points Tracker v1.0")

# Initialize the main window
root = tk.Tk()
root.title("Sort Bot Points Tracker")
root.geometry("400x300")
root.configure(bg="lightgreen")

# Create menu bar
menu_bar = tk.Menu(root)

# Add file menu
file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Log Out", command=logout)
file_menu.add_command(label="Quit", command=quit_app)

# Add edit menu
edit_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Edit", menu=edit_menu)
edit_menu.add_command(label="Increase Points", command=increase_points)
edit_menu.add_command(label="Decrease Points", command=decrease_points)
edit_menu.add_command(label="Reset Points", command=reset_points)

# Add help menu
help_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Help", menu=help_menu)
help_menu.add_command(label="About", command=show_about)

# Login frame
login_frame = tk.Frame(root, bg="lightgreen")
tk.Label(login_frame, text="Username:", font=("Helvetica", 14), bg="lightgreen").pack(pady=10)
username_entry = tk.Entry(login_frame, font=("Helvetica", 14))
username_entry.pack(pady=10)
username_entry.bind("<Return>", handle_enter)
tk.Label(login_frame, text="Password:", font=("Helvetica", 14), bg="lightgreen").pack(pady=10)
password_entry = tk.Entry(login_frame, font=("Helvetica", 14), show="*")
password_entry.pack(pady=10)
password_entry.bind("<Return>", handle_enter)
tk.Button(login_frame, text="Login", command=login, bg="lightblue", font=("Helvetica", 14)).pack(pady=10)
tk.Button(login_frame, text="Register", command=register, bg="lightyellow", font=("Helvetica", 14)).pack(pady=10)
login_frame.pack()

# Main frame
main_frame = tk.Frame(root, bg="lightgreen")
tk.Button(main_frame, text="Log Out", command=logout, bg="lightgrey", font=("Helvetica", 14)).pack(pady=10)
points_label = tk.Label(main_frame, text="Points: 0", font=("Helvetica", 16), bg="lightgreen")
points_label.pack(pady=20)
tk.Button(main_frame, text="Increase Points", command=increase_points, bg="lightblue", font=("Helvetica", 14)).pack(pady=10)
tk.Button(main_frame, text="Decrease Points", command=decrease_points, bg="lightyellow", font=("Helvetica", 14)).pack(pady=10)
tk.Button(main_frame, text="Reset Points", command=reset_points, bg="lightcoral", font=("Helvetica", 14)).pack(pady=10)

# Start the GUI event loop
root.mainloop()

# Close the database connection when the application is closed
conn.close()
