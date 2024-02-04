import cv2 as cv
import math
import numpy as np
from matplotlib import pyplot as plt


# JPEG quantification matrix
# receives:
#   - boolean to specify is Y or Cr/Cb component
#   - compression factor (0..100)

def getQuantificationMatrix(LuminanceOrChrominance, compfactor):
    lumQuant = [16, 11, 10, 16, 24, 40, 51, 61,
                12, 12, 14, 19, 26, 58, 60, 55,
                14, 13, 16, 24, 40, 57, 69, 56,
                14, 17, 22, 29, 51, 87, 80, 62,
                18, 22, 37, 56, 68, 109, 103, 77,
                24, 35, 55, 64, 81, 104, 113, 92,
                49, 64, 78, 87, 103, 121, 120, 101,
                72, 92, 95, 98, 112, 100, 103, 99]

    ChrQuant = [17, 18, 24, 47, 99, 99, 99, 99,
                18, 21, 26, 66, 99, 99, 99, 99,
                24, 26, 56, 99, 99, 99, 99, 99,
                47, 66, 99, 99, 99, 99, 99, 99,
                99, 99, 99, 99, 99, 99, 99, 99,
                99, 99, 99, 99, 99, 99, 99, 99,
                99, 99, 99, 99, 99, 99, 99, 99,
                99, 99, 99, 99, 99, 99, 99, 99
                ]

    matrix = np.zeros((8, 8), dtype=float)
    idx = 0
    Quant = lumQuant if LuminanceOrChrominance else ChrQuant
    for y in range(0, 8):
        for x in range(0, 8):
            matrix[y, x] = Quant[idx] * 100.0 / compfactor
            idx += 1

    return matrix


def blockProcessing_compress(imgChannelBlock, luminanceOrChrominance, compFactor):
    # b)	Convert block to float format and subtract the DC component (128)
    imgChannelBlock = np.float32(imgChannelBlock) - 128
    # c)	Apply the Discrete Cosine Transform (DCT)
    imgChannelBlock_dct = cv.dct(imgChannelBlock)
    # d)	Coefficients Quantization (divide by quantification matrix and round)
    quantificationMatrix = getQuantificationMatrix(luminanceOrChrominance, compFactor)
    divided_img = imgChannelBlock_dct / quantificationMatrix
    # e)	Coefficients rounding (math.round)
    result = np.round(divided_img)

    return result


def blockProcessing_decompress(result, luminanceOrChrominance, compFactor):
    # f) Coefficients recovering
    result = np.multiply(result, getQuantificationMatrix(luminanceOrChrominance, compFactor))
    # g)	Apply the Discrete Cosine Inverse Transform
    result = cv.dct(result, None, flags=cv.DCT_INVERSE)
    # h) Add DC component, clip to 0..255 and convert to byte
    result = np.clip(result + 128, 0, 255).astype(np.uint8)

    return result
