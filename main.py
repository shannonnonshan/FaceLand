# main.py
import numpy as np
import ttkbootstrap as tb
import tkinter as tk
from tkinter import Canvas
import cv2
import customtkinter as ctk
from PIL import Image, ImageTk
from welcome_page import show_welcome_page
ctk.deactivate_automatic_dpi_awareness()

# Khởi tạo cửa sổ chính
root = tb.Window(themename="superhero")
root.title("Funny Face Filters")
root.geometry("800x700")
window_width = 800
window_height = 700


screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2

root.geometry(f"{window_width}x{window_height}+{x}+{y}")
show_welcome_page(root)

root.mainloop()






