# main.py
import tkinter as tk
from tkinter import Canvas
import cv2
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
show_welcome_page(root)

root.mainloop()
