import tkinter as tk
from tkinter import filedialog
import cv2 as cv
from imageForms import showSideBySideImages
import numpy as np


def Temp_Match_Alg():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    imgOriginal = cv.imread(file_path)

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    template = cv.imread(file_path)

    result = cv.matchTemplate(imgOriginal, template, cv.TM_SQDIFF)
    MinVal, MaxVal, MinLoc, MaxLoc = cv.minMaxLoc(result)
    color1 = (0, 255, 0)
    color2 = (255, 0, 0)
    cv.drawMarker(result, MinLoc, color1)
    cv.drawMarker(result, MaxLoc, color2)

    showSideBySideImages(imgOriginal, result, "Images", False, False)


def Viola_Jones():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    imgOriginal = cv.imread(file_path)

    haar = cv.CascadeClassifier(r"C:/Users/jcinv/PycharmProjects/TAPDI/Aula09/models/haarcascade_frontalface_alt2.xml")
    width = np.int32(imgOriginal.shape[0])
    height = np.int32(imgOriginal.shape[1])
    faces = haar.detectMultiScale(
        imgOriginal, scaleFactor=1.15,
        minSize=(20, 20),
        maxSize=(width // 2, height // 2))

    for (x, y, w, h) in faces:
        img = cv.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 0, 255), 4)

    showSideBySideImages(img, imgOriginal, "Images", False, False)


def Hog_Alg():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    imgOriginal = cv.imread(file_path)

    hog = cv.HOGDescriptor()
    hog.setSVMDetector(cv.HOGDescriptor.getDefaultPeopleDetector())
    pedestrians = hog.detectMultiScale(imgOriginal, scale = 1.005)
    for (x, y, w, h) in pedestrians[0]:
        img = cv.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 0, 255), 4)

    showSideBySideImages(img, imgOriginal, "Images", False, False)