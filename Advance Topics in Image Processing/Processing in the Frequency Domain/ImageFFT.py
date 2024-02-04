
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import imageForms as iF
import math

#DLL TO CALL C#
import sys
import clr


def GetFFT_Mag_Phase(imgGray):
    """
    Get FFT magnitude an phase
    :param imgGray:
    :return: Magnitude and Phase
    """

    rows, cols = imgGray.shape
    m = cv.getOptimalDFTSize( rows )
    n = cv.getOptimalDFTSize( cols )
    imgGray_padded = cv.copyMakeBorder(imgGray, 0, m - rows, 0, n - cols, cv.BORDER_CONSTANT, value=[0, 0, 0])
    imgGray_padded = np.float32(imgGray_padded)
    planes = [np.float32(imgGray_padded), np.zeros(imgGray_padded.shape, np.float32)]
    imgComplex_planes = cv.merge(planes)  # Add to the expanded another plane with zeros

    imgComplex_planes = cv.dft(imgGray_padded,flags=cv.DFT_COMPLEX_OUTPUT )

    cv.split(imgComplex_planes, planes)  # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    imgMag = np.zeros(imgGray_padded.shape, np.float32)
    imgPhase = np.zeros(imgGray_padded.shape, np.float32)
    cv.cartToPolar(planes[0], planes[1], imgMag, imgPhase)
    return imgMag, imgPhase


def GetFFT_Inverse_Mag_Phase(MagInv, PhaseInv):
    """
    Get inverse FFT
    :param Mag:
    :param Phase2:
    :return: image
    """
    complexX = np.zeros(MagInv.shape,np.float32)
    complexY = np.zeros(MagInv.shape, np.float32)
    cv.polarToCart(MagInv,PhaseInv, complexX, complexY)

    planes = [complexX, complexY]
    complexI = cv.merge(planes)  # Add to the expanded another plane with zeros

    complexI= cv.idft(complexI, flags=cv.DFT_SCALE)

    cv.split(complexI, planes)  # planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    imgRes = planes[0]
    cv.normalize(imgRes,imgRes,0,255,cv.NORM_MINMAX)
    return imgRes


def CreateFilterMask_Ideal(shape, cutOff, HighPass):
    """
    generate a ideal filter mask
    :param shape:
    :param cutOff:
    :param HighPass:
    :return:
    """
    shk = np.int32 (shape[0]/2),np.int32( shape[1]/2)

    if (HighPass):
        mask = np.ones(shape, np.float32)
        cv.circle(mask,(shk),cutOff,0,cv.FILLED)
    else:
        mask = np.zeros(shape, np.float32)
        cv.circle(mask,(shk),cutOff,1,cv.FILLED)

    return mask

def CreateFilterMask_Gaussian(shape, cutOff, HighPass):
    """
    generate a gaussian filter mask
    :param shape:
    :param cutOff:
    :param HighPass:
    :return:
    """
    shk = np.int32 (shape[0]/2),np.int32( shape[1]/2)
    width, height = shape
    mask = np.zeros(shape, np.float32)
    order = 3
    for u in range(0, width):
        for v in range(0, height):
            uu = u - width / 2.0
            vv = v - height / 2.0
            if (HighPass):
                mask[u, v] = 1 - 1.0 / (1 + ((math.sqrt(2) - 1) * math.pow(math.sqrt(uu * uu + vv * vv) / (cutOff), 2*order)))

#                mask[u, v] = 1 - 1.0 * math.exp(-math.sqrt(uu * uu + vv * vv) / (2 * cutOff * cutOff))
            else:
                mask[u, v] = 1.0 / (1+ ((math.sqrt(2)-1) * math.pow( math.sqrt(uu * uu + vv * vv) / (cutOff), 2 *order )))
 #               mask[u, v] = 1.0 * math.exp(-math.sqrt(uu * uu + vv * vv) / (2 * cutOff * cutOff))

    return mask

# ############################################################################
# ############################################################################
# ############################################################################

#Open image
pathname = "..\\Aula 5 - FFT\\images\\"

