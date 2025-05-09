import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
import customtkinter as ctk

import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import Canvas
from PIL import Image, ImageTk

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('./haarcascade/haarcascade_mcs_nose.xml')

# Load filter images
glasses = cv2.imread("filters/glasses.png", cv2.IMREAD_UNCHANGED)
hat = cv2.imread("filters/hat.png", cv2.IMREAD_UNCHANGED)
mustache = cv2.imread("filters/mustache.png", cv2.IMREAD_UNCHANGED)

def overlay_image(bg, overlay, x, y, size):
    w, h = size
    if w <= 0 or h <= 0:
        return bg
    overlay = cv2.resize(overlay, (w, h))
    if overlay.shape[2] == 3:
        alpha_channel = np.ones((h, w, 1), dtype=overlay.dtype) * 255
        overlay = np.concatenate([overlay, alpha_channel], axis=2)
    b, g, r, a = cv2.split(overlay)
    mask = cv2.merge((a, a, a))
    if x < 0 or y < 0 or x + w > bg.shape[1] or y + h > bg.shape[0]:
        return bg
    roi = bg[y:y+h, x:x+w]
    bg_part = cv2.bitwise_and(roi.copy(), 255 - mask)
    fg_part = cv2.bitwise_and(overlay[:, :, :3], mask)
    result = cv2.add(bg_part, fg_part)
    bg[y:y+h, x:x+w] = result
    return bg

# Initialize tkinter window
root = tb.Window(themename="superhero")
root.title("Funny Face Filters")
root.geometry("800x650")
window_width = 800
window_height = 650

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2

root.geometry(f"{window_width}x{window_height}+{x}+{y}")
#root.resizable(False, False)


# ========== Welcome Page ==========
# Load images
bg_image = Image.open("background.jpg").resize((800, 650))
bg_photo = ImageTk.PhotoImage(bg_image)

team_img = Image.open("team_photo.png").resize((350, 250))
team_photo = ImageTk.PhotoImage(team_img)

# Welcome Frame
welcome_frame = tb.Frame(root)
welcome_frame.pack(fill="both", expand=True)

bg_canvas = Canvas(welcome_frame, width=800, height=800, bg="#75d0ef")
bg_canvas.pack(fill="both", expand=True)
bg_canvas.create_image(0, 0, image=bg_photo, anchor="nw")

# Labels
#subjectName = ["Subject Name"]
canvas_width = 800
title_y = 30
subtitle_y = 60

bg_canvas.create_text(canvas_width // 2, title_y,
                      text="Subject Name",
                      font=("Helvetica", 14, "bold"),
                      fill="darkblue",
                      anchor="n")  # anchor 'n' để canh từ trên xuống

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

canvas_width = 800
canvas_height = 600

start_x = 160
gap = 250
y_position = canvas_height - 30  # nằm gần mép dưới

for idx, name in enumerate(students):
    x = start_x + idx * gap
    y = y_position

    # Lớp bóng màu trắng (lệch nhẹ)
    bg_canvas.create_text(x + 2, y + 2,
                          text=name,
                          font=("Helvetica", 12, "bold"),
                          fill="white",
                          anchor="s")

    # Lớp chính màu xanh lá
    bg_canvas.create_text(x, y,
                          text=name,
                          font=("Helvetica", 12, "bold"),
                          fill="#5B913B",
                          anchor="s")
bg_canvas.create_image(400, 280, image=team_photo)

# Start Button
def start_app():
    welcome_frame.pack_forget()
    main_frame.pack(fill="both", expand=True)
    update_frame()
   
   # Cấu hình canvas để hỗ trợ transparency
bg_canvas.configure(bg='#75d0ef')  # Màu nền canvas
   
   # Đặt nút lên canvas (cách tốt hơn)

start_button = ctk.CTkButton(master=bg_canvas,
                             text="Let's Get Started",
                             font=("Helvetica", 20, "bold"),
                             width=200,
                             height=60,
                             corner_radius=20,
                             fg_color="#FE7743",
                             text_color="white",         # Chữ trắng
                             hover_color="#3E4A6C",      # Màu khi hover
                             command=start_app)

# Đặt nút lên canvas tại vị trí mong muốn
bg_canvas.create_window(400, 400, window= start_button)

# ========== Main Page ==========
main_frame = tb.Frame(root)

title_label = tb.Label(main_frame, text="🎭 Funny Face Filters 🎭", font=("Helvetica", 22, "bold"), bootstyle="inverse-info")
title_label.pack(pady=10)

canvas = Canvas(main_frame, width=640, height=420)
canvas.pack()

filter_frame = tb.Frame(main_frame)
filter_frame.pack(pady=15)

glasses_button = ctk.CTkButton(master=filter_frame,
                               text="Glasses",
                               command=lambda: select_filter("glasses"),
                               fg_color="black",
                               text_color="white", width=55, height=35,
                               font=("Arial", 12,"bold"),
                               corner_radius=20)
glasses_button.pack(side=ctk.LEFT, padx=10)

hat_button = ctk.CTkButton(master=filter_frame,
                           text="Hat",
                           command=lambda: select_filter("hat"),
                           fg_color="black",
                           font=("Arial", 12,"bold"),
                           text_color="white", width=55, height=35,
                           corner_radius=20)
hat_button.pack(side=ctk.LEFT, padx=10)
mustache_button = ctk.CTkButton(master=filter_frame,
                               text="Mustache",
                               command=lambda: select_filter("mustache"),
                               fg_color="black",
                               font=("Arial", 12,"bold"),
                               text_color="white", width=55, height=35,
                               corner_radius=20)
mustache_button.pack(side=ctk.LEFT, padx=10)

# Theme switcher
def change_theme(theme_name):
    root.style.theme_use(theme_name)

themes = ["superhero", "darkly", "cosmo", "morph", "flatly"]
theme_menu = tb.Menubutton(filter_frame, text="Change Theme", bootstyle="secondary outline") # Place in filter_frame
menu = tb.Menu(theme_menu)
theme_menu["menu"] = menu
for t in themes:
    menu.add_command(label=t, command=lambda name=t: change_theme(name))
theme_menu.pack(side=LEFT, padx=10) # Use pack with side=LEFT
# Capture button
def capture_image():
    ret, frame = cap.read()
    if ret:
        filename = "captured_image.png"
        cv2.imwrite(filename, frame)
        print(f"Image saved as {filename}")

camera_icon = ctk.CTkImage(
            light_image=Image.open("cameraicon.png"),
            size=(30, 30)
        )

capture_button = ctk.CTkButton(
    master=main_frame,
    text="",
    image = camera_icon,
    command=capture_image,
    width=60,
    height=60,
    corner_radius=60,  # Bằng 1/2 chiều cao để tạo hình tròn
    fg_color="#FF6363",
    hover_color="#3CB371"
)
capture_button.pack(pady=15)

# Webcam logic
cap = cv2.VideoCapture(0)
current_filter = None

def select_filter(filter_name):
    global current_filter
    current_filter = filter_name

def update_frame():
    global current_filter, cap
    ret, frame = cap.read()
    if not ret:
        return
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (fx, fy, fw, fh) in faces:
        roi_gray = gray[fy:fy+fh, fx:fx+fw]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if current_filter == "glasses" and len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda x: x[0])
            (ex1, ey1, ew1, eh1) = eyes[0]
            (ex2, ey2, ew2, eh2) = eyes[1]
            eye1_center = (fx + ex1 + ew1 // 2, fy + ey1 + eh1 // 2)
            eye2_center = (fx + ex2 + ew2 // 2, fy + ey2 + eh2 // 2)
            dx = eye2_center[0] - eye1_center[0]
            glasses_width = int(2.2 * abs(dx))
            glasses_height = int(glasses_width * glasses.shape[0] / glasses.shape[1])
            center_x = (eye1_center[0] + eye2_center[0]) // 2
            center_y = (eye1_center[1] + eye2_center[1]) // 2
            x = center_x - glasses_width // 2
            y = center_y - glasses_height // 2
            frame = overlay_image(frame, glasses, x, y, (glasses_width, glasses_height))

        elif current_filter == "hat":
            hat_width = fw
            hat_height = int(hat_width * hat.shape[0] / hat.shape[1])
            hx = fx
            hy = fy - hat_height + 15
            frame = overlay_image(frame, hat, hx, hy, (hat_width, hat_height))

        elif current_filter == "mustache":
            nose = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)
            if len(nose) > 0:
                nx, ny, nw, nh = nose[0]
                mustache_width = int(nw * 1.5)
                mustache_height = int(mustache_width * mustache.shape[0] / mustache.shape[1])
                mx = fx + nx + nw // 2 - mustache_width // 2
                my = fy + ny + nh - 40
                frame = overlay_image(frame, mustache, mx, my, (mustache_width, mustache_height))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img)
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image = img_tk
    canvas.after(10, update_frame)

# Run the GUI
root.mainloop()
cap.release()
cv2.destroyAllWindows()
