# welcome_page.py
import ttkbootstrap as tb
from tkinter import Canvas
from PIL import Image, ImageTk
import customtkinter as ctk
from main_page import show_main_page
import threading

def show_welcome_page(root):
    # Load hình ảnh
    bg_image = Image.open("images/background.jpg").resize((800, 650))  # Resize to 800x650
    bg_photo = ImageTk.PhotoImage(bg_image)  # Convert to Tkinter-compatible format

    team_img = Image.open("images/team_photo.png").resize((350, 250))
    team_photo = ImageTk.PhotoImage(team_img)

    # Khung chứa màn hình chào mừng
    welcome_frame = tb.Frame(root)
    welcome_frame.pack(fill="both", expand=True)

    # Create canvas with the correct size to match the image
    bg_canvas = Canvas(welcome_frame, width=800, height=650, bg="#75d0ef")
    bg_canvas.pack(fill="both", expand=True)
    
    bg_canvas.create_image(0, 0, image=bg_photo, anchor="nw")
    bg_canvas.bg_photo = bg_photo

    # Labels
    canvas_width = 800
    title_y = 30
    subtitle_y = 60

    bg_canvas.create_text(canvas_width // 2, title_y,
                          text="Subject Name",
                          font=("Helvetica", 14, "bold"),
                          fill="darkblue",
                          anchor="n")

    bg_canvas.create_text(canvas_width // 2, subtitle_y,
                          text="Digital Image Processing",
                          font=("Helvetica", 16,"bold"),
                          fill="#3E7B27",
                          anchor="n")
    bg_canvas.create_text(700, 100,
                          text="Dr. Hoàng Văn Dũng",
                          font=("Helvetica", 12,"bold"),
                          fill="black",
                          anchor="n")

    students = ["Đinh Thị Thanh Vy  22110093", 
                "Đoàn Minh Khanh   22110042", 
                "Lê Thị Thu Hương  22110040"]

    start_x = 160
    gap = 250
    y_position = 570

    for idx, name in enumerate(students):
        x = start_x + idx * gap
        y = y_position

        bg_canvas.create_text(x + 2, y + 2,
                              text=name,
                              font=("Helvetica", 12, "bold"),
                              fill="white",
                              anchor="s")

        bg_canvas.create_text(x, y,
                              text=name,
                              font=("Helvetica", 12, "bold"),
                              fill="#5B913B",
                              anchor="s")
    
    bg_canvas.create_image(400, 280, image=team_photo)
    bg_canvas.team_photo = team_photo
    bg_canvas.configure(bg='#75d0ef')
    def start_app():
        welcome_frame.pack_forget()
        show_main_page(root)
    start_button = ctk.CTkButton(master=bg_canvas,
                                text="Let's Get Started",
                                font=("Helvetica", 20, "bold"),
                                width=200,
                                height=60,
                                corner_radius=20,
                                fg_color="#FE7743",
                                text_color="white",
                                hover_color="#3E4A6C",
                                command=start_app)

    bg_canvas.create_window(400, 400, window=start_button)

    # # Keep a reference to the background image to avoid garbage collection
    # bg_canvas.image = bg_photo

    
