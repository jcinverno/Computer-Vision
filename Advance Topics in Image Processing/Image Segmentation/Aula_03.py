import tkinter as tk
from tkinter import filedialog
from imageForms import showSideBySideImages
from ImageSegmentation import GetConnectedComponents
from ImageSegmentation import Kmeans_Clustering
from ImageSegmentation import GetWatershedFromMarks, GetWatershedByImmersion
from ImageSegmentation import GPLSegmentation

import cv2 as cv

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

img = cv.imread(file_path)


def show_components(imgOriginal):
    imgLabels = GetConnectedComponents(imgOriginal)
    imgOriginal[imgLabels == 0] = [255, 0, 0]
    showSideBySideImages(imgOriginal, imgLabels, title="Connected Components", BGR1=False, BGR2=False)


def show_kmeans(img, k):
    imgK = Kmeans_Clustering(img, k)
    showSideBySideImages(img, imgK, title="K-Means", BGR1=False, BGR2=False)


def show_watershed_withMarks(img):
    labels = cv.imread(file_path + ".mask.bmp")
    withMarks = GetWatershedFromMarks(img, labels)
    showSideBySideImages(img, withMarks, title="Watershed_Marks", BGR1=False, BGR2=False)


def show_watershed_withNoMarks(img):
    noMarks = GetWatershedByImmersion(img)
    showSideBySideImages(img, noMarks, title="Watershed_No_Marks", BGR1=False, BGR2=False)


def show_GPLSegmentation(img):
    GPLSegmentation(img)


show_watershed_withNoMarks(img)