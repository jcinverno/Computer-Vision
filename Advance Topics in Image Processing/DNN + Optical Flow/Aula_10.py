import cv2
import numpy as np
from imageOpticalFlow import *
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from PIL import Image


def getFaceBox(frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    # Open DNN model
    modelFile = r"C:/Users/jcinv/PycharmProjects/TAPDI/Aula10/models/opencv_face_detector_uint8.pb"
    configFile = r"C:/Users/jcinv/PycharmProjects/TAPDI/Aula10/models/opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
    # prepare for DNN
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (500, 300), [104,
                                                                   117, 123], True, False)
    # set image as DNN input
    net.setInput(blob)
    # get Output
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, bboxes


def lookForFacesWebcam():

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frameOpencvDnn, bboxes = getFaceBox(frame)
        cv2.imshow('Face Detection', frameOpencvDnn)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def opticalFlow(LucasKanade, threshold=6):

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()

    if LucasKanade:
        LucasKanade_OF(file_path, threshold)

    else:
        Farneback_OF(file_path)


def predict_digit():

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.resize(img, (28, 28))
    img = img / 255.0
    img = np.reshape(img, (1, 28, 28))

    model = tf.keras.models.load_model("mnist_model.h5")
    predictions = model.predict(img)
    predicted_digit = np.argmax(predictions)

    return predicted_digit


print(predict_digit())

#lookForFacesWebcam()
#opticalFlow(False, 5)



