import tensorflow as tf
import numpy as np
import cv2 as cv
from keras.models import Sequential,load_model
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense


# Cargar los pesos
model=load_model('BatchNormalizeFacialKeyPointsModel.h5')

# Iniciar la captura de vídeo
cap = cv.VideoCapture(0)

# Iniciar el detector de caras
faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")# haarcascade_eye_tree_eyeglasses.xml

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # Convertir a escala de grises

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    faces = faceCascade.detectMultiScale(frame_gray, minNeighbors=5, scaleFactor=1.3, minSize=(30, 30))
    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        roi_resized = cv.resize(roi_color, (96, 96))
        roi_resized_gray = cv.cvtColor(roi_resized, cv.COLOR_BGR2GRAY)

        roi_resized_gray = np.expand_dims(roi_resized_gray, axis=-1)
        keypoints = model.predict(np.expand_dims(roi_resized_gray, axis=0))

        for i in range(0, len(keypoints[0]), 2):
            x_point = int(keypoints[0][i])+x
            y_point = int(keypoints[0][i+1])+y
            print("X:",x_point)
            print("y:", y_point)
            # Dibujar los puntos faciales
            cv.circle(frame, (x_point, y_point), 3, (0, 0, 255), -1)

    cv.imshow('color', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()