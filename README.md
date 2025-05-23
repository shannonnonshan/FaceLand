
**Author:** Group 11

1. Doan Minh Khanh 22110042
2. Le Thi Thu Huong 22110040
3. Dinh Thi Thanh Vy 22110093

---

# Face Filter App

### Description

This is a Python application using **OpenCV** and **Tkinter (CustomTkinter + TTKBootstrap)** that allows users to apply **virtual glasses, hats, or mustaches** directly onto their face in real time using a webcam. The app supports fun photo capturing and filter customization.

### Technologies Used

- **Python 3.8+**
- **OpenCV** – Face, eye, and nose detection
- **Tkinter + ttkbootstrap + customtkinter** – GUI framework
- **Pillow (PIL)** – Image processing
- **NumPy** – Image data manipulation

---

### Features

- Live webcam feed with real-time face detection  
- Virtual glasses applied based on eye position  
- Virtual hats applied with smooth tracking on head  
- Virtual mustache detection based on nose location  
- User-friendly interface with clickable filter buttons  
- Add custom filters through a dedicated interface  
- Capture and save photos to the `capture` folder  

---

## Project Structure

project_root/
│

├── main_page.py # Main GUI and webcam logic

├── addfilter.py # GUI for adding new filters

├── filters/ # Filter image folders

│ ├── glasses/ # Glasses images (PNG with transparency)

│ ├── hats/ # Hat images

│ └── mustaches/ # Mustache images

├── haarcascade/ # Haar cascade classifiers

│ └── haarcascade_mcs_nose.xml

├── images/ # GUI icons (optional)

├── capture/ # Saved photos

└── README.md # Project documentation

---

### How to Run

 1. Install Required Libraries

```bash
pip install opencv-python pillow numpy ttkbootstrap customtkinter
```

 2. Install Required Libraries

```bash
python main.py
```

### Note

- All filters must be PNG images with transparent backgrounds.
- A working webcam is required.
- Large images may affect performance.
- You can add new filters using the "Add Filter" button in the app.