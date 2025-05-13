# face_detection.py

from __future__ import print_function
import cv2 as cv

def detect_faces(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x,y,w,h) in faces:
        center = (x + w//2, y + h//2)
        frame = cv.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        faceROI = frame_gray[y:y+h,x:x+w]
        eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(faceROI)
        for (x2,y2,w2,h2) in eyes:
            eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            radius = int(round((w2 + h2)*0.25))
            frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4)
    return frame

def apply_filter_to_face(image, filter_image, face_coordinates):
    """
    Hàm áp dụng filter lên khuôn mặt nhận diện được.
    """
    (x, y, w, h) = face_coordinates
    filter_resized = cv.resize(filter_image, (w, h))
    
    for i in range(h):
        for j in range(w):
            if filter_resized[i, j][3] > 0:  # Kiểm tra alpha channel của filter
                image[y + i, x + j] = filter_resized[i, j][:3]
    
    return image
