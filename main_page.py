# main_page.py
import numpy as np
import ttkbootstrap as tb
from tkinter import Canvas
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
current_glasses_index = 0
current_hat_index = 0
current_mustache_index = 0
current_filter = None
def show_main_page(root):
    global canvas, cap  # cho phép update_frame truy cập được

    main_frame = tb.Frame(root)
    main_frame.pack()

    canvas = Canvas(main_frame, width=640, height=380)
    canvas.pack(pady=25)
    # Webcam logic
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_mcs_nose.xml')

    # Load filter images
    glasses = [
        cv2.imread("filters/glasses.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("filters/glass2.png", cv2.IMREAD_UNCHANGED)
    ]

    hats = [
        cv2.imread("filters/hat.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("filters/hat2.png", cv2.IMREAD_UNCHANGED)
    ]

    mustaches = [
        cv2.imread("filters/mustache.png", cv2.IMREAD_UNCHANGED),
        cv2.imread("filters/mustache2.png", cv2.IMREAD_UNCHANGED)
    ]

   
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
    


    filter_frame = tb.Frame(main_frame)
    filter_frame.pack(pady=15)

    # Glasses buttons
    def next_glasses():
        global current_glasses_index
        current_glasses_index = (current_glasses_index + 1) % len(glasses)
        select_filter("filters/glasses")

    glass_icon1 = ctk.CTkImage(
        light_image=Image.open("filters/glasses.png"),
        size=(35, 35)
    )

    glass_icon2 = ctk.CTkImage(
        light_image=Image.open("filters/glass2.png"),
        size=(35, 35)
    )

    glasses_button1 = ctk.CTkButton(master=filter_frame,
                                text="",
                                image=glass_icon1,
                                command=lambda: [select_filter("glasses"), set_glasses_index(0)],
                                fg_color="#EFEEEA",
                                border_width=2,
                                border_color="#222222",
                                text_color="white", width=50, height=25,
                                font=("Arial", 12,"bold"),
                                corner_radius=100)
    glasses_button1.pack(side=ctk.LEFT, padx=5)

    glasses_button2 = ctk.CTkButton(master=filter_frame,
                                text="",
                                image=glass_icon2,
                                command=lambda: [select_filter("glasses"), set_glasses_index(1)],
                                fg_color="#EFEEEA",
                                border_width=2,
                                border_color="#222222",
                                text_color="white", width=50, height=25,
                                font=("Arial", 12,"bold"),
                                corner_radius=100)
    glasses_button2.pack(side=ctk.LEFT, padx=5)

    # Hat buttons
    def next_hat():
        global current_hat_index
        current_hat_index = (current_hat_index + 1) % len(hats)
        select_filter("filters/hat")

    hat_icon1 = ctk.CTkImage(
        light_image=Image.open("filters/hat.png"),
        size=(35, 35)
    )

    hat_icon2 = ctk.CTkImage(
        light_image=Image.open("filters/hat2.png"),
        size=(35, 35)
    )

    hat_button1 = ctk.CTkButton(master=filter_frame,
                            text="",
                            image=hat_icon1,
                            command=lambda: [select_filter("hat"), set_hat_index(0)],
                            fg_color="#EFEEEA",
                            border_width=2,
                            border_color="#222222",
                            font=("Arial", 12,"bold"),
                            text_color="white", width=50, height=25,
                            corner_radius=25)
    hat_button1.pack(side=ctk.LEFT, padx=5)

    hat_button2 = ctk.CTkButton(master=filter_frame,
                            text="",
                            image=hat_icon2,
                            command=lambda: [select_filter("hat"), set_hat_index(1)],
                            fg_color="#EFEEEA",
                            border_width=2,
                            border_color="#222222",
                            font=("Arial", 12,"bold"),
                            text_color="white", width=50, height=25,
                            corner_radius=25)
    hat_button2.pack(side=ctk.LEFT, padx=5)

    # Mustache buttons
    def next_mustache():
        global current_mustache_index
        current_mustache_index = (current_mustache_index + 1) % len(mustaches)
        select_filter("filters/mustache")

    mustache_icon1 = ctk.CTkImage(
        light_image=Image.open("filters/mustache.png"),
        size=(35, 35)
    )

    mustache_icon2 = ctk.CTkImage(
        light_image=Image.open("filters/mustache2.png"),
        size=(35, 35)
    )

    mustache_button1 = ctk.CTkButton(master=filter_frame,
                                text="",
                                image=mustache_icon1,
                                command=lambda: [select_filter("mustache"), set_mustache_index(0)],
                                fg_color="#EFEEEA",
                                border_width=2,
                                border_color="#222222",
                                font=("Arial", 12,"bold"),
                                text_color="white", width=50, height=25,
                                corner_radius=25)
    mustache_button1.pack(side=ctk.LEFT, padx=5)

    mustache_button2 = ctk.CTkButton(master=filter_frame,
                                text="",
                                image=mustache_icon2,
                                command=lambda: [select_filter("mustache"), set_mustache_index(1)],
                                fg_color="#EFEEEA",
                                border_width=2,
                                border_color="#222222",
                                font=("Arial", 12,"bold"),
                                text_color="white", width=50, height=25,
                                corner_radius=25)
    mustache_button2.pack(side=ctk.LEFT, padx=5)

    def set_glasses_index(index):
        global current_glasses_index
        current_glasses_index = index

    def set_hat_index(index):
        global current_hat_index
        current_hat_index = index

    def set_mustache_index(index):
        global current_mustache_index
        current_mustache_index = index

    # Theme switcher
    # def change_theme(theme_name):
    #     root.style.theme_use(theme_name)

    # themes = ["superhero", "darkly", "cosmo", "morph", "flatly"]
    # theme_menu = tb.Menubutton(filter_frame, text="Change Theme", bootstyle="secondary outline")
    # menu = tb.Menu(theme_menu)
    # theme_menu["menu"] = menu
    # for t in themes:
    #     menu.add_command(label=t, command=lambda name=t: change_theme(name))
    # theme_menu.pack(side=LEFT, padx=10)

    # Capture button
    def capture_image():
        ret, frame = cap.read()
        if ret:
            filename = "captured_image.png"
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")

    camera_icon = ctk.CTkImage(
        light_image=Image.open("images/cameraicon.png"),
        size=(30, 30)
    )

    capture_button = ctk.CTkButton(
        master=main_frame,
        text="",
        image=camera_icon,
        command=capture_image,
        width=60,
        height=60,
        border_width=0,
        corner_radius=60,
        fg_color="#FF6363",
        hover_color="#3CB371"
    )
    capture_button.pack(pady=40)

    
   

    def select_filter(filter_name):
        global current_filter
        current_filter = filter_name

    def update_frame():
        global current_filter, cap, current_glasses_index, current_hat_index, current_mustache_index
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
                glasses_height = int(glasses_width * glasses[current_glasses_index].shape[0] / glasses[current_glasses_index].shape[1])
                center_x = (eye1_center[0] + eye2_center[0]) // 2
                center_y = (eye1_center[1] + eye2_center[1]) // 2
                x = center_x - glasses_width // 2
                y = center_y - glasses_height // 2
                frame = overlay_image(frame, glasses[current_glasses_index], x, y, (glasses_width, glasses_height))

            elif current_filter == "hat":
                hat_width = fw
                hat_height = int(hat_width * hats[current_hat_index].shape[0] / hats[current_hat_index].shape[1])
                hx = fx
                hy = fy - hat_height + 15
                frame = overlay_image(frame, hats[current_hat_index], hx, hy, (hat_width, hat_height))

            elif current_filter == "mustache":
                nose = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)
                if len(nose) > 0:
                    nx, ny, nw, nh = nose[0]
                    mustache_width = int(nw * 1.5)
                    mustache_height = int(mustache_width * mustaches[current_mustache_index].shape[0] / mustaches[current_mustache_index].shape[1])
                    mx = fx + nx + nw // 2 - mustache_width // 2
                    my = fy + ny + nh - 40
                    frame = overlay_image(frame, mustaches[current_mustache_index], mx, my, (mustache_width, mustache_height))

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor="nw", image=img_tk)
        canvas.image = img_tk
        canvas.after(10, update_frame)

    update_frame()
    # root.mainloop()
    # cap.release()
    # cv2.destroyAllWindows()



