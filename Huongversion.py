import cv2
import numpy as np
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import Canvas
from PIL import Image, ImageTk
import ctypes
awareness = ctypes.c_int()
try:
    print("DPI awareness level:", awareness.value)
    ctypes.windll.shcore.SetProcessDpiAwareness(1)  # System DPI aware
    print("DPI awareness level:", awareness.value)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()
# Kiểm tra DPI awareness
print("DPI awareness level:", awareness.value)
# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('./haarcascade/haarcascade_mcs_nose.xml')

# Load filter images (with alpha channel)
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

# Initialize the tkinter window
root = tb.Window(themename="superhero")
root.title("Funny Face Filters")
root.geometry("800x650")
window_width = 800
window_height = 700

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2

root.geometry(f"{window_width}x{window_height}+{x}+{y}")
root.resizable(False, False)

# Title label
title_label = tb.Label(root, text="🎭 Funny Face Filters 🎭", font=("Helvetica", 24, "bold"), bootstyle="inverse-info")
title_label.pack(pady=10)

# Canvas for webcam
canvas = Canvas(root, width=640, height=480)
canvas.pack()

# Capture from camera
cap = cv2.VideoCapture(0)
current_filter = None

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

        if current_filter == "glasses":
            if len(eyes) >= 2:
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

def select_filter(filter_name):
    global current_filter
    current_filter = filter_name

def capture_image():
    ret, frame = cap.read()
    if ret:
        filename = "captured_image.png"
        cv2.imwrite(filename, frame)
        print(f"Image saved as {filename}")

# Filter buttons frame
filter_frame = tb.Frame(root)
filter_frame.pack(pady=10)

glasses_button = tb.Button(filter_frame, text="Glasses", command=lambda: select_filter("glasses"), bootstyle="primary")
glasses_button.grid(row=0, column=0, padx=10)

hat_button = tb.Button(filter_frame, text="Hat", command=lambda: select_filter("hat"), bootstyle="info")
hat_button.grid(row=0, column=1, padx=10)

mustache_button = tb.Button(filter_frame, text="Mustache", command=lambda: select_filter("mustache"), bootstyle="warning")
mustache_button.grid(row=0, column=2, padx=10)

# Theme changer
def change_theme(theme_name):
    root.style.theme_use(theme_name)

themes = ["superhero", "darkly", "cosmo", "morph", "flatly"]
theme_menu = tb.Menubutton(root, text="Change Theme", bootstyle="secondary outline")
menu = tb.Menu(theme_menu)
theme_menu["menu"] = menu

for t in themes:
    menu.add_command(label=t, command=lambda name=t: change_theme(name))

theme_menu.pack(pady=10)

# Capture button
capture_button = tb.Button(root, text="📸 Capture Image", command=capture_image, bootstyle="success", width=25)
capture_button.pack(pady=15)

# Start loop
update_frame()
root.mainloop()
cap.release()
cv2.destroyAllWindows()