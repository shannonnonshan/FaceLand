import os
import shutil
from tkinter import Canvas, filedialog, messagebox
from PIL import Image, ImageTk
import ttkbootstrap as tb
import customtkinter as ctk
import threading
def show_add_filter_page(root):
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

    # Action function (generalized)
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

            if width != height :
                messagebox.showerror("Error", "The picture must square shape.")
                return

            if img.mode != "RGBA" or not img.getchannel("A").getextrema()[0] < 255:
                messagebox.showerror("Error", "Transparent image (PNG RGBA).")
                return

            folder = os.path.join("filters", filter_type)
            os.makedirs(folder, exist_ok=True)
            dest_path = os.path.join(folder, os.path.basename(filepath))
            shutil.copy(filepath, dest_path)

            messagebox.showinfo("Success","Successfully Added")

        except Exception as e:
            messagebox.showerror("Error", f"Cancel handle this image:\n{str(e)}")

    # Buttons
    button_style = {
        "font": ("Helvetica", 16, "bold"),
        "width": 200,
        "height": 50,
        "corner_radius": 20,
        "fg_color": "#FE7743",
        "text_color": "white",
        "hover_color": "#3E4A6C",
    }
    def go_back():
        from main_page import show_main_page
        add_filter_frame.pack_forget()  # Ẩn frame hiện tại
        threading.Thread(target=show_main_page, args=(root,), daemon=True).start()
    add_glasses_btn = ctk.CTkButton(master=bg_canvas,
                                    text="Add Glasses",
                                    command=lambda: select_filter_image("glasses"),
                                    **button_style)

    add_hat_btn = ctk.CTkButton(master=bg_canvas,
                                text="Add Hat",
                                command=lambda: select_filter_image("hats"),
                                **button_style)

    add_mustache_btn = ctk.CTkButton(master=bg_canvas,
                                     text="Add Mustache",
                                     command=lambda: select_filter_image("mustaches"),
                                     **button_style)
    back_button = ctk.CTkButton(master=bg_canvas,
                            text="← Back to Main Page",
                            font=("Helvetica", 14),
                            width=180,
                            height=40,
                            corner_radius=15,
                            fg_color="#3E4A6C",
                            text_color="white",
                            hover_color="#5C6B8A",
                            command=go_back)

    # Place buttons
    bg_canvas.create_window(400, 450, window=add_glasses_btn)
    bg_canvas.create_window(250, 520, window=add_hat_btn)
    bg_canvas.create_window(550, 520, window=add_mustache_btn)
    bg_canvas.create_window(90, 600, window=back_button)
