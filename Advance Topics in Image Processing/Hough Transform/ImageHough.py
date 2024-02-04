import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import imageForms as iF
import math


def HoughPlane(img, minAngle, maxAngle, angleSpacing):
    """
    Hough plane --
    adapted from https://alyssaq.github.io/2014/understanding-hough-transform/

    :param img: img must be single channel
    :return: accumulator, thetas, rhos
    """

    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(minAngle, maxAngle, angleSpacing))
    width, height = img.shape
    diag_len = np.uint32(np.ceil(np.sqrt(width * width + height * height)))  # max_dist
    rhos = np.linspace(-diag_len / 2, diag_len / 2, diag_len)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((diag_len, num_thetas), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges

    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = round((x - width / 2) * cos_t[t_idx] + (y - height / 2) * sin_t[t_idx] + diag_len / 2)
            accumulator[rho, t_idx] += 1

    return accumulator, thetas, rhos


def ShowHoughLines(img, imgOriginal, thresh):
    """
    Draw Hough lines openCV
    :param img: gray scale 
    :return: img containing lines
    """

    lines = cv.HoughLines(img, 1, np.pi / 180, thresh)
    imgOriginalRes = np.copy(imgOriginal)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv.line(imgOriginalRes, pt1, pt2, (0, 0, 255), 1, cv.LINE_AA)

    return imgOriginalRes


def ShowHoughLineSegments(img, imgOriginal, thresh):
    """
    Draw Hough lines openCV
    :param img: gray scale
    :return: img containing lines
    """
    linesP = cv.HoughLinesP(img, 1, np.pi / 180, thresh)
    imgOriginalRes = np.copy(imgOriginal)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(imgOriginalRes, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 1, cv.LINE_AA)

    return imgOriginalRes


def ShowHoughCircles(img, imgOriginal):
    """
    Draw Hough lines openCV
    :param img: gray scale
    :return: img containing lines
    """
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT,1, 50, param1=140, param2=120, minRadius=30, maxRadius=120)
    imgOriginalRes = np.copy(imgOriginal)

    if circles is not None:
        for i in range(0, len(circles[0])):
            center = (np.int32(circles[0][i][0]), np.int32(circles[0][i][1]))
            radius = np.int32(circles[0][i][2])

            # circle outline
            cv.circle(imgOriginalRes, center, radius, (0, 0, 255), 1)

    return imgOriginalRes


def ShowVideo(filename):
    """
    Play a video with OpenCV
    :param filename:
    :return:
    """
    vidCap = cv.VideoCapture(filename)

    low_white = np.array([0, 0, 174])
    up_white = np.array([179, 18, 255])

    if (not vidCap.isOpened()):
        print("Video File Not Found")
        exit(-1)

    while (True):
        ret, vidFrame = vidCap.read()
        if (not ret):
            break

        hsv = cv.cvtColor(vidFrame, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, low_white, up_white)

        edges = cv.Canny(mask, 75, 150)
        lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv.line(vidFrame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        cv.imshow("frame", mask)
        key = cv.waitKey(1)
        if key == 27:
            break
    cv.destroyAllWindows()


