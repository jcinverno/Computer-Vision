import tkinter as tk
from tkinter import filedialog
from imageForms import showSideBySideImages
from ImageHough import HoughPlane
from ImageHough import ShowHoughLineSegments
from ImageHough import ShowHoughLines
from ImageHough import ShowHoughCircles
from ImageHough import ShowVideo
import cv2 as cv
import numpy as np

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()
img = cv.imread(file_path)


def show_accumulator(img):
    accumulator = HoughPlane(cv.cvtColor(img, cv.COLOR_BGR2GRAY), 0, 180, 1)
    showSideBySideImages(img, accumulator[0], False, False)


def show_houghLines(img, threshold):
    imgHough = ShowHoughLines(cv.cvtColor(img, cv.COLOR_BGR2GRAY), img, threshold)
    showSideBySideImages(img, imgHough)


def hough_real_img(img, threshold):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgSobel = cv.Sobel(imgGray, cv.CV_8U, 1, 0)
    _, bin_img = cv.threshold(imgSobel, 255, 255, cv.THRESH_OTSU + cv.THRESH_BINARY, imgSobel)
    imgHough = ShowHoughLines(bin_img, img, threshold)
    showSideBySideImages(imgHough, imgSobel, False, False)


# hough_real_img(img, 120)

def hough_segments_canny(img, threshold):
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgCanny = cv.Canny(imgGray, 100, 50)
    imgHough = ShowHoughLineSegments(imgCanny, img, threshold)
    showSideBySideImages(imgHough, imgCanny, False, False)


# hough_segments_canny(img, 50)

def hough_circles(img, threshold):
    img_circles = ShowHoughCircles(cv.cvtColor(img, cv.COLOR_BGR2GRAY), img, threshold)
    showSideBySideImages(img, img_circles, False, False)


#hough_circles(img)
ShowVideo(file_path)