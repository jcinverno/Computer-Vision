import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import imageForms as iF

# DLL TO CALL C#
import sys
import clr

# DLL import
sys.path.append(r".\GPL_LIB")
clr.AddReference('GPL_lib')
from GPL_lib import GPL_lib


def GetConnectedComponents(img):
    """
    Connected Components segmentation
    :param img: input image (BGR)
    :return: outputs image with contours
    """
    # prepare image for CC
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    retVal, img_thresh = cv.threshold(imgGray, 0, 255, cv.THRESH_OTSU + cv.THRESH_BINARY)

    # CC
    retVal, labels = cv.connectedComponents(img_thresh)

    labels_uint8 = np.uint8(labels + 1)

    # Find contours
    contours, hierarchy = cv.findContours(img_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Draw contours
    for i in range(len(contours)):
        color = (0, 0, 0)
        cv.drawContours(labels_uint8, contours, i, color, 1, cv.LINE_8, hierarchy, 0)

    return labels_uint8


def GetWatershedFromMarks(img, imgLabels):
    """

    :param img: input image (BGR)
    :param imgLabels: initial labels image (BGR)
    :return: segmented image
    """

    # prepare image markers
    # must be grayscale
    imgLabels = cv.cvtColor(imgLabels, cv.COLOR_BGR2GRAY)
    imgLabels = np.int32(imgLabels)
    iF.showSideBySideImages(img, imgLabels)

    # apply watershed
    imgWatershed = cv.watershed(img, imgLabels)

    return imgWatershed + 1


def GetWatershedByImmersion(img):
    """
    Watershed by Immersion from minimum values
    :param image: input image (BGR)
    :return: segmented image
    """
    from Watershed import Watershed

    w = Watershed()
    # expects gray scale image
    imgGray = cv.cvtColor(255 - img, cv.COLOR_BGR2GRAY)
    labels = w.apply(imgGray)

    return labels


def GPLSegmentation(image):
    """
    Get GPL segmentation and show configuration form
    :param image: input image (BGR)
    :return: segmented image
    """
    my_instance = GPL_lib(image.tobytes(), image.shape[0], image.shape[1], True)
    my_instance.ShowConfigForm()
    buffer = my_instance.Get_OutputImage()
    imgOut = np.frombuffer(buffer, dtype=np.uint8)
    deserialized_x = np.reshape(imgOut, newshape=(image.shape[0], image.shape[1]))
    return deserialized_x


def Kmeans_Clustering(image, k):
    """
    Apply opencv kmeans clustering
    :param image: input image (BGR)
    :param k: number of clusters
    :return: segmented image
    """

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # apply Kmeans  with 10 attemps. The attempt with best compactness is returned
    _, labels, (centers) = cv.kmeans(pixel_values, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels]

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image
