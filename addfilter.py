import os
import shutil
from tkinter import Canvas, filedialog, messagebox
from PIL import Image, ImageTk
import ttkbootstrap as tb
import customtkinter as ctk

def show_add_filter_page(root):
    team_img = Image.open("images/team_photo.png").resize((350, 250))
    team_photo = ImageTk.PhotoImage(team_img)

    add_filter_frame = tb.Frame(root)
    add_filter_frame.pack(fill="both", expand=True)

    bg_canvas = Canvas(add_filter_frame, width=800, height=650, bg="#75d0ef")
    bg_canvas.pack(fill="both", expand=True)

    canvas_width = 800
    title_y = 30
    subtitle_y = 60

    bg_canvas.create_text(canvas_width // 2, title_y,
                          text="Add Filter Page",
                          font=("Helvetica", 18, "bold"),
                          fill="darkblue",
                          anchor="n")

    bg_canvas.create_text(canvas_width // 2, subtitle_y,
                          text="Upload PNG Filter (Max 250x250, Transparent BG)",
                          font=("Helvetica", 14),
                          fill="#3E7B27",
                          anchor="n")

    bg_canvas.create_image(400, 280, image=team_photo)
    bg_canvas.team_photo = team_photo

    # Action function
    def select_filter_image():
        filepath = filedialog.askopenfilename(
            title="Chọn ảnh filter",
            filetypes=[("PNG Images", "*.png")]
        )
        if not filepath:
            return
        
        try:
            img = Image.open(filepath)
            width, height = img.size

            if width != height or width > 250:
                messagebox.showerror("Lỗi", "Ảnh phải là hình vuông và ≤ 250x250 pixels.")
                return

            if img.mode != "RGBA" or not img.getchannel("A").getextrema()[0] < 255:
                messagebox.showerror("Lỗi", "Ảnh phải có nền trong suốt (PNG RGBA).")
                return

            os.makedirs("filters", exist_ok=True)
            dest_path = os.path.join("filters", os.path.basename(filepath))
            shutil.copy(filepath, dest_path)

            messagebox.showinfo("Success", f"Add filter: {os.path.basename(filepath)}")

        except Exception as e:
            messagebox.showerror("Error", f"Cancel handle thí image:\n{str(e)}")

    # Nút chọn ảnh
    add_button = ctk.CTkButton(master=bg_canvas,
                               text="Pick a remove background picture",
                               font=("Helvetica", 16, "bold"),
                               width=200,
                               height=50,
                               corner_radius=20,
                               fg_color="#FE7743",
                               text_color="white",
                               hover_color="#3E4A6C",
                               command=select_filter_image)

    bg_canvas.create_window(400, 450, window=add_button)
