# face_detection.py

import cv2

def detect_faces(image):
    """
    Hàm nhận diện khuôn mặt trong ảnh sử dụng Haar Cascade.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return faces

def apply_filter_to_face(image, filter_image, face_coordinates):
    """
    Hàm áp dụng filter lên khuôn mặt nhận diện được.
    """
    (x, y, w, h) = face_coordinates
    filter_resized = cv2.resize(filter_image, (w, h))
    
    for i in range(h):
        for j in range(w):
            if filter_resized[i, j][3] > 0:  # Kiểm tra alpha channel của filter
                image[y + i, x + j] = filter_resized[i, j][:3]
    
    return image
