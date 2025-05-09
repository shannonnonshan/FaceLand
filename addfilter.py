import os
import shutil
from tkinter import Canvas, filedialog, messagebox
from PIL import Image, ImageTk
import ttkbootstrap as tb
import customtkinter as ctk
import threading

def show_add_filter_page(root):
    # Load ảnh nhóm và background
    team_img = Image.open("images/team_photo.png").resize((350, 250))
    team_photo = ImageTk.PhotoImage(team_img)
    
    # Load background image và resize cho phù hợp với kích thước cửa sổ
    bg_image = Image.open("images/pinkpastel.jpg")
    bg_photo = ImageTk.PhotoImage(bg_image.resize((800, 650)))

    # Tạo Frame và Canvas
    add_filter_frame = tb.Frame(root)
    add_filter_frame.pack(fill="both", expand=True)

    bg_canvas = Canvas(add_filter_frame, width=800, height=650, highlightthickness=0)
    bg_canvas.pack(fill="both", expand=True)
    
    # Thêm background image vào canvas
    bg_canvas.create_image(0, 0, image=bg_photo, anchor="nw")
    bg_canvas.bg_photo = bg_photo  # Giữ tham chiếu ảnh

    # Tiêu đề và mô tả
    canvas_width = 800
    title_y = 30
    subtitle_y = 60

    bg_canvas.create_text(canvas_width // 2, title_y,
                        text="Add Filter Page",
                        font=("Helvetica", 18, "bold"),
                        fill="white",
                        anchor="n")

    bg_canvas.create_text(canvas_width // 2, subtitle_y,
                        text="Upload PNG Filter (Max 250x250, Transparent BG)",
                        font=("Helvetica", 14),
                        fill="#F7FFFE",
                        anchor="n")

    bg_canvas.create_image(400, 200, image=team_photo)
    bg_canvas.team_photo = team_photo  # Giữ tham chiếu ảnh

    # Hàm chọn và lưu filter
    def select_filter_image(filter_type):
        filepath = filedialog.askopenfilename(
            title="Chọn ảnh filter",
            filetypes=[("PNG Images", "*.png")]
        )
        if not filepath:
            return

        try:
            img = Image.open(filepath)
            width, height = img.size

            if width != height:
                messagebox.showerror("Error", "The picture must be square shape.")
                return

            if img.mode != "RGBA" or not img.getchannel("A").getextrema()[0] < 255:
                messagebox.showerror("Error", "Transparent image (PNG RGBA).")
                return

            folder = os.path.join("filters", filter_type)
            os.makedirs(folder, exist_ok=True)
            dest_path = os.path.join(folder, os.path.basename(filepath))
            shutil.copy(filepath, dest_path)

            messagebox.showinfo("Success", "Successfully Added")

        except Exception as e:
            messagebox.showerror("Error", f"Cannot handle this image:\n{str(e)}")

    # Quay về trang chính
    def go_back():
        from main_page import show_main_page
        add_filter_frame.pack_forget()
        threading.Thread(target=show_main_page, args=(root,), daemon=True).start()

    # Style nút
# Tạo một frame trung gian trong suốt để chứa các nút
# Tạo một frame trung gian trong suốt để chứa các nút
    button_container = ctk.CTkFrame(
        bg_canvas,
        fg_color="#88ccfc",
        bg_color="#88ccfc",
        width=800,
        height=650
    )
    bg_canvas.create_window(400, 390, window=button_container)  # Đặt ở giữa canvas

    # Style nút
    button_style = {
        "font": ("Helvetica", 16, "bold"),
        "width": 200,
        "height": 50,
        "corner_radius": 20,
        "fg_color": "#FE7743",
        "text_color": "white",
        "hover_color": "#3E4A6C",
        "bg_color": "transparent",
        "border_width": 0
    }

    # Nút Add Hat ở giữa trên cùng
    add_hat_btn = ctk.CTkButton(
        master=button_container,
        text="Add Hat",
        command=lambda: select_filter_image("hats"),
        **button_style
    )
    add_hat_btn.pack(pady=(50, 20))  # pady=(top, bottom)

    # Frame cho 2 nút dưới
    bottom_buttons_frame = ctk.CTkFrame(
        button_container,
        fg_color="transparent",
        bg_color="transparent"
    )
    bottom_buttons_frame.pack(pady=10)

    # Nút Add Glasses (bên trái)
    add_glasses_btn = ctk.CTkButton(
        master=bottom_buttons_frame,
        text="Add Glasses",
        command=lambda: select_filter_image("glasses"),
        **button_style
    )
    add_glasses_btn.pack(side="left", padx=20)

    # Nút Add Mustache (bên phải)
    add_mustache_btn = ctk.CTkButton(
        master=bottom_buttons_frame,
        text="Add Mustache",
        command=lambda: select_filter_image("mustaches"),
        **button_style
    )
    add_mustache_btn.pack(side="left", padx=20)

    # Nút Back ở dưới cùng
    back_button = ctk.CTkButton(
        master=button_container,
        text="Back to Main Page",
        font=("Helvetica", 16, "bold"),
        width=180,
        height=50,
        corner_radius=10,
        fg_color="#FEBA17",
        text_color="white",
        hover_color="#5C6B8A",
        bg_color="transparent",
        border_width=0,
        command=go_back
    )
    back_button.pack(pady=(30, 20))    # bg_canvas.create_window(400, 350, window=add_glasses_btn)
    # bg_canvas.create_window(250, 420, window=add_hat_btn)
    # bg_canvas.create_window(550, 420, window=add_mustache_btn)
    # bg_canvas.create_window(400, 500, window=back_button)