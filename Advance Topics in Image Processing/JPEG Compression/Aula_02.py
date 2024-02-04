import tkinter as tk
from tkinter import filedialog

import cv2
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

from JPEGCompression import blockProcessing_compress, blockProcessing_decompress

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

img = cv.imread(file_path)


def codification(img, luminanceOrChrominance , compFactor):
    compressed_blocks = []
    img = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    imgChannelY, imgChannelCb, imgChannelCr = cv.split(img)
    height, width = img.shape[:2]
    for e in range(0, height, 8):
        for f in range(0, width, 8):
            startX, endX, startY, endY = e, e + 8, f, f + 8
            imgChannelBlock = imgChannelY[startX:endX, startY:endY]
            result = blockProcessing_compress(imgChannelBlock, luminanceOrChrominance, compFactor)

            result = blockProcessing_decompress(result, luminanceOrChrominance, compFactor)
            imgChannelY[startX:endX, startY:endY] = result

    result = cv.merge((imgChannelY, imgChannelCb, imgChannelCr))
    result = cv.cvtColor(result, cv.COLOR_YCrCb2BGR)

    return result


compressed_image = codification(img, True, 30)

# Set the figure size for displaying images
plt.figure(figsize=(12, 6))  # Adjust the size as needed

# Display the original and compressed images side by side
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(compressed_image, cv.COLOR_BGR2RGB))
plt.title("Compressed Image")
plt.axis('off')

# Show the side-by-side images
plt.show()
cv.waitKey(0)