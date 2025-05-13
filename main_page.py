# main_page.py
import numpy as np
import ttkbootstrap as tb
from tkinter import Canvas
import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import threading
from addfilter import show_add_filter_page
import os
from datetime import datetime
import csv
import time
import psutil
from face_detection import detect_faces, apply_filter_to_face

processed_frame = None
prev_hat_pos = None
prev_face = None

current_glasses_index = 0
current_hat_index = 0
current_mustache_index = 0
current_filter = None
current_fps = 0
prev_time = time.time()
running = True
logging_started = False
fps_buffer = []
def start_logging_thread():
    global logging_started
    if logging_started:
            return
    logging_started = True
    def log_performance():
        while running:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cpu_percent = psutil.cpu_percent(interval=1)  # interval=1 để lấy mẫu ổn định
            ram_percent = psutil.virtual_memory().percent

            # Lấy giá trị FPS mới nhất từ biến toàn cục
            global current_fps
            fps = current_fps

            with open("fps_cpu_data.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, fps, cpu_percent, ram_percent])

            time.sleep(1.5)  # log mỗi ~2 giây

    t = threading.Thread(target=log_performance, daemon=True)
    t.start()
def show_main_page(root):
    start_logging_thread()
    global canvas, cap, running, logging_started
    global filter_buttons_frame
    cap = None
    running = True
    logging_started = False
    main_frame = tb.Frame(root)
    main_frame.pack()

    canvas = Canvas(main_frame, width=640, height=380)
    canvas.pack(pady=25)
   
    # Webcam logic
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_mcs_nose.xml')
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        tb.Messagebox.show_error("Webcam Error", "Could not open webcam.")
        return
    # Load filter images
    filter_types = ["glasses", "hats", "mustaches"]
    filters = {ftype: [] for ftype in filter_types}

    for ftype in filter_types:
        folder_path = f"filters/{ftype}"
        for idx, filename in enumerate(sorted(os.listdir(folder_path))):
            if filename.endswith(".png"):
                path = os.path.join(folder_path, filename)
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                filters[ftype].append({"img": img, "path": path, "index": idx})
    def select_filter(filter_name):
        global current_filter
        current_filter = filter_name
    def set_filter_index(filter_type, index):
        global current_glasses_index, current_hat_index, current_mustache_index
        if filter_type == "glasses":
            current_glasses_index = index
        elif filter_type == "hats":
            current_hat_index = index
        elif filter_type == "mustaches":
            current_mustache_index = index
    def create_filter_buttons(filter_type, frame):
        for item in filters[filter_type]:
            icon = ctk.CTkImage(
                light_image=Image.open(item["path"]),
                size=(35, 35)
            )
            btn = ctk.CTkButton(master=frame,
                                text="",
                                image=icon,
                                command=lambda i=item["index"]: [
                                    select_filter(filter_type),
                                    set_filter_index(filter_type, i)
                                ],
                                fg_color="#EFEEEA",
                                border_width=2,
                                border_color="#222222",
                                text_color="white", width=50, height=25,
                                font=("Arial", 12, "bold"),
                                corner_radius=100)
            btn.pack(side=ctk.LEFT, padx=5)
   
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
    feature_frame = tb.Frame(main_frame)
    feature_frame.pack(pady=15)  
    # Capture button
    def capture_image():
        global processed_frame
        if processed_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}.png"

            os.makedirs("capture", exist_ok=True)
            filepath = os.path.join("capture", filename)

            cv2.imwrite(filepath, processed_frame)
            print(f"Image saved as {filepath}")
        else:
            print("No processed frame available to save.")

    camera_icon = ctk.CTkImage(
        light_image=Image.open("images/cameraicon.png"),
        size=(30, 30)
    )

    capture_button = ctk.CTkButton(
        master=feature_frame,
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
    capture_button.pack(side=ctk.LEFT, padx=5)
    def add_filter():
        global running
        running = False
        main_frame.pack_forget()
        show_add_filter_page(root)

    add_filter_icon = ctk.CTkImage(
        light_image=Image.open("images/add.png"),
        size=(40, 40)
    )

    add_filter_button = ctk.CTkButton(master=feature_frame,
                                text="",
                                image=add_filter_icon,
                                command=add_filter,
                                fg_color="#E9A319",
                                # border_width=2,
                                # border_color="#222222",
                                font=("Arial", 12,"bold"),
                                text_color="white", width=62, height=62,
                                corner_radius=60)
    add_filter_button.pack(side=ctk.LEFT, padx=5)
    def update_scroll_region(event):
        filter_frame.configure(scrollregion=filter_frame.bbox("all"))

    back_icon = ctk.CTkImage(
    light_image=Image.open("images/icon_back.png"),
    size=(40, 40))

    forward_icon = ctk.CTkImage(
        light_image=Image.open("images/icon_forward.png"),
        size=(40, 40))
    
    filter_container = tb.Frame(main_frame)
    filter_container.pack(fill="x", pady=5)

    scroll_frame = tb.Frame(filter_container)
    scroll_frame.pack(fill="x")

    left_btn = ctk.CTkButton(
        scroll_frame,
        text="",
        image=back_icon,
        command=lambda: filter_frame.xview_scroll(-1, "units"),
        width=40,
        height=40,
        fg_color="transparent",
        hover_color="#FFC1B4",
        corner_radius=10
    )
    left_btn.pack(side="left", padx=5)

    filter_frame = Canvas(scroll_frame, height=120, highlightthickness=0)
    filter_frame.pack(side="left", fill="x", expand=True)

    right_btn = ctk.CTkButton(
        scroll_frame,
        text="",
        image=forward_icon,
        command=lambda: filter_frame.xview_scroll(1, "units"),
        width=40,
        height=40,
        fg_color="transparent",
        hover_color="#FFC1B4",
        corner_radius=11
    )
    right_btn.pack(side="left", padx=5)

    filter_buttons_frame = tb.Frame(filter_frame)
    filter_frame.create_window((0, 0), window=filter_buttons_frame, anchor="nw")
    create_filter_buttons("glasses", filter_buttons_frame)
    create_filter_buttons("hats", filter_buttons_frame)
    create_filter_buttons("mustaches", filter_buttons_frame)
    filter_buttons_frame.bind("<Configure>", update_scroll_region)
    
    def update_frame():
        global prev_time, current_fps, avg_fps
        global fps_buffer
        global current_filter, cap, current_glasses_index, current_hat_index, current_mustache_index
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                tb.Messagebox.show_error("Webcam Error", "Could not open webcam.")
                return
        ret, frame = cap.read()
        if not ret:
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        detected_faces = []
        global prev_face
        
        for (fx, fy, fw, fh) in faces:
            roi_gray = gray[fy:fy+fh, fx:fx+fw]
            detected_faces.append((fx, fy, fw, fh))

            if current_filter == "glasses":
                eyes = eye_cascade.detectMultiScale(roi_gray)
                if len(eyes) >= 2:
                    eyes = sorted(eyes, key=lambda x: x[0])
                    (ex1, ey1, ew1, eh1) = eyes[0]
                    (ex2, ey2, ew2, eh2) = eyes[1]
                    eye1_center = (fx + ex1 + ew1 // 2, fy + ey1 + eh1 // 2)
                    eye2_center = (fx + ex2 + ew2 // 2, fy + ey2 + eh2 // 2)
                    dx = eye2_center[0] - eye1_center[0]
                    glasses_width = int(2.2 * abs(dx))
                    img = filters["glasses"][current_glasses_index]["img"]
                    glasses_height = int(glasses_width * img.shape[0] / img.shape[1])
                    center_x = (eye1_center[0] + eye2_center[0]) // 2
                    center_y = (eye1_center[1] + eye2_center[1]) // 2
                    x = center_x - glasses_width // 2
                    y = center_y - glasses_height // 2
                    frame = overlay_image(frame, img, x, y, (glasses_width, glasses_height))

            elif current_filter == "hats":
                hat_width = int(fw*1.2)
                imgh = filters["hats"][current_hat_index]["img"]
                hat_height = int(hat_width * imgh.shape[0] / imgh.shape[1])
                hx = fx - int((hat_width - fw) / 2)
                hy = fy - hat_height + 30
                frame = overlay_image(frame, imgh, hx, hy, (hat_width, hat_height))

            elif current_filter == "mustaches":
                nose = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)
                if len(nose) > 0:
                    nx, ny, nw, nh = nose[0]
                    mustache_width = int(nw * 1.5)
                    imgm = filters["mustaches"][current_mustache_index]["img"]
                    mustache_height = int(mustache_width * imgm.shape[0] / imgm.shape[1])
                    mx = fx + nx + nw // 2 - mustache_width // 2
                    my = fy + int(fh*0.55)
                    frame = overlay_image(frame, imgm, mx, my, (mustache_width, mustache_height))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        end = time.time()
        current_fps = 1 / (end - prev_time)
        prev_time = end
        

        # Trong update_frame():
        fps_buffer.append(current_fps)
        if len(fps_buffer) > 100:
            fps_buffer.pop(0)
        avg_fps = sum(fps_buffer) / len(fps_buffer)
        # Lấy thông tin CPU và RAM
        cpu_percent = psutil.cpu_percent(interval=None)  # không chờ
        ram_percent = psutil.virtual_memory().percent

        # Vẽ các thông số lên frame
        cv2.putText(frame, f"FPS: {current_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (144, 238, 144), 2)  # Light Green
        cv2.putText(frame, f"CPU: {cpu_percent:.0f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 102), 2)  # Light Yellow
        cv2.putText(frame, f"RAM: {ram_percent:.0f}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 102, 102), 2)  # Light Red

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        global processed_frame
        processed_frame = frame.copy()
        img = Image.fromarray(frame_rgb)
        img_tk = ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor="nw", image=img_tk)
        canvas.image = img_tk
        canvas.after(10, update_frame)

    update_frame()
    # root.mainloop()
    # cap.release()
    # cv2.destroyAllWindows()