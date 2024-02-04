import tkinter as tk
from tkinter import filedialog

import cv2
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import huffman as h

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

img = cv.imread(file_path)
cv.imshow("Imagem", img)


def plot_hist(img):
    # Calculate the histogram
    hist = cv.calcHist([img], [0], None, [256], [0, 256])

    # Plot the histogram
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.title("Histogram")
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.show()


def print_pixel_count(img):
    # Calculate the histogram
    hist = cv.calcHist([img], [0], None, [256], [0, 256])

    # Create a dictionary to store pixel values and their counts
    pixel_counts = {}
    unique_values = np.unique(img)
    total_pixels = img.shape[0] * img.shape[1]

    # Create empty lists for grouped values and percentages
    grouped_values = []
    grouped_percentages = []

    for i in range(len(unique_values)):
        value = unique_values[i]
        frequency = hist[value][0]

        grouped_values.append(f"{value}")
        grouped_percentages.append((frequency / total_pixels) * 100)

    # Print the grouped histogram in terms of percentages
    for i in range(len(grouped_values)):
        print(f"{grouped_values[i]} - {grouped_percentages[i]:.2f}%")


def huffman_formula(img):
    huffman_dict = h.huffman(img.flatten())
    return huffman_dict


def huffman_codification(img):
    encoded = ''
    huffman_dict = h.huffman(img.flatten())
    encoded += format(len(huffman_dict), '08b')
    for key in huffman_dict:
        encoded += huffman_dict[key].zfill(8)
    for pixel in img.flatten():
        encoded += huffman_dict[pixel]
    return encoded


def compression_ration(img):

    height, width, channel = img.shape
    simple_rep = width * height * channel

    encoded = len(huffman_codification(img))

    ratio = encoded / simple_rep
    return ratio


def entropy_image(img):
    blue_channel, green_channel, red_channel = cv2.split(img)

    hist_blue = cv2.calcHist([blue_channel], [0], None, [256], [0, 256])
    hist_green = cv2.calcHist([green_channel], [0], None, [256], [0, 256])
    hist_red = cv2.calcHist([red_channel], [0], None, [256], [0, 256])

    prob_blue = hist_blue / np.sum(hist_blue)
    prob_green = hist_green / np.sum(hist_green)
    prob_red = hist_red / np.sum(hist_red)

    entropy_blue = -np.sum(prob_blue * np.log2(prob_blue + np.finfo(float).eps))
    entropy_green = -np.sum(prob_green * np.log2(prob_green + np.finfo(float).eps))
    entropy_red = -np.sum(prob_red * np.log2(prob_red + np.finfo(float).eps))

    print(f"Red Channel Entropy: {round(float(entropy_red), 2)}")
    print(f"Green Channel Entropy: {round(float(entropy_green), 2)}")
    print(f"Blue Channel Entropy: {round(float(entropy_blue), 2)}")


blurred_image = cv2.GaussianBlur(img, (3, 3), 0)
entropy_image(blurred_image)
#entropy_image(img)

cv.waitKey(0)
