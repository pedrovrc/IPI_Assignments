#   Nome: Pedro Victor Rodrigues de Carvalho
#   Matrícula: 17/0113043
#   Universidade de Brasília, 2º semestre de 2018
#   Curso: Engenharia de Computação
#   Matéria: Introdução ao Processamento de Imagens
#   Professor: Alexandre Zaghetto

#   Primeira parte do Assignment 2: Filtragem no domínio espacial e da frequência
#   Linguagem de programação: Python 3

import cv2
import numpy as np
import glob


# ---------- Start of functions ----------

def bgr2ycbcr(image):  # Converts image from bgr space to ycbcr space
    height, width, channels = image.shape

    Y = np.zeros((height, width), dtype=np.uint8)
    Y_double = np.zeros((height, width), dtype=np.double)

    # Y_double made to facilitate fast processing while avoiding rounding errors
    Y_double[:, :] += image[:, :, 2] * 0.299
    Y_double[:, :] += image[:, :, 1] * 0.587
    Y_double[:, :] += image[:, :, 0] * 0.114

    Y[:, :] = Y_double[:, :]

    Cb = np.zeros((height, width), dtype=np.uint8)
    Cr = np.zeros((height, width), dtype=np.uint8)

    # Other channels derived from Y channel, blue and red
    Cb[:, :] = (0.564 * image[:, :, 0]) - (0.564 * Y[:, :]) + 128
    Cr[:, :] = (0.713 * image[:, :, 2]) - (0.713 * Y[:, :]) + 128

    return [Y, Cb, Cr]


def ycbcr2bgr(Y, Cb, Cr):  # Converts image from ycbcr space to bgr space
    height, width = Y.shape
    red = np.zeros((height, width), dtype=np.uint8)
    green = np.zeros((height, width), dtype=np.uint8)
    blue = np.zeros((height, width), dtype=np.uint8)

    # bgr channels are derived from ycbcr channels
    blue[:, :] = Y[:, :] + ((1.772 * Cb[:, :]) - (128 * 1.772))
    green[:, :] = Y[:, :] - ((0.344136 * Cb[:, :]) - (128 * 0.344136)) - ((0.714136 * Cr[:, :]) - (128 * 0.714136))
    red[:, :] = Y[:, :] + ((1.402 * Cr[:, :]) - (128 * 1.402))

    # Channels are concatenated into final product image
    result = np.zeros((height, width, 3), dtype=np.uint8)
    result[:, :, 0] = blue[:, :]
    result[:, :, 1] = green[:, :]
    result[:, :, 2] = red[:, :]

    return result


def get_pixels(image, i, j):  # Returns a pixel and its 8 neighbours in an array (used in salt and pepper filter)
    px1 = image[i, j]
    px2 = image[i-1, j-1]
    px3 = image[i, j-1]
    px4 = image[i+1, j-1]
    px5 = image[i+1, j]
    px6 = image[i+1, j+1]
    px7 = image[i, j+1]
    px8 = image[i-1, j+1]
    px9 = image[i-1, j]
    return [px1, px2, px3, px4, px5, px6, px7, px8, px9]


def low_pass_cylinder(image, radius):  # Simple low pass ideal filter
    height, width = image.shape
    for i in range(0, height):
        for j in range(0, width):
            if radius*radius <= ((height/2) - i)*((height/2) - i) + ((height/2) - j)*((height/2) - j):
                image[i, j] = 0


def gaussian_filter(image):  # Uses all images in directory to filter using the average of pixels
    # Load all images in directory in grayscale mode (only Y channel)
    img_array = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob('Images/*.bmp')]
    img_number = len(img_array)

    height, width = image.shape
    sum = np.zeros((height, width), dtype=np.uint16)  # 16-bit integer used to prevent overflowing

    # For all images, add the Y channel to the sum matrix
    for k in range(0, img_number):
        sum[:, :] += img_array[k][:, :]

    # Divide each pixel value by the total number of images and save result
    image[:, :] = sum[:, :] / img_number


def salt_pepper_filter(image):  # Uses median to remove peaks of 255 or 0 in image
    height, width = image.shape

    # Create padding around image
    borderedImage = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # For each pixel, get it and its neighbours in an array and get the median value as the new value
    for i in range(0, height):
        for j in range(0, width):
            pixel_array = get_pixels(borderedImage, i+1, j+1)
            pixel_array.sort()
            image[i, j] = pixel_array[4]


def frequency_filter(image):  # Uses fourier's transform to filter patterns in the image
    # Process the shifted transform
    transf_shift = np.fft.fftshift(np.fft.fft2(image))

    # Process spectrum image
    magnitude_spectrum = 20*np.log(np.abs(transf_shift))
    val_max = magnitude_spectrum.max()
    magnitude_spectrum[:, :] = magnitude_spectrum[:, :] / val_max
    cv2.imshow('magnitude spectrum', magnitude_spectrum)
    cv2.waitKey(0)

    # Apply low pass ideal filter to transform
    low_pass_cylinder(transf_shift, 100)

    # Process new magnitude spectrum for visualisation
    magnitude_spectrum = 20 * np.log(np.abs(transf_shift))
    val_max = magnitude_spectrum.max()
    magnitude_spectrum[:, :] = magnitude_spectrum[:, :] / val_max
    cv2.imshow('magnitude spectrum filtered', magnitude_spectrum)
    cv2.waitKey(0)

    # Process inverse transform and shift and save in image
    image[:, :] = np.uint8(np.abs(np.fft.ifft2(np.fft.ifftshift(transf_shift[:, :]))))


# ---------- End of functions ----------
# ---------- Start of main program ----------

# Read and show original image
rgbsample = cv2.imread("Images/1.bmp")
cv2.imshow("BGR Original", rgbsample)
cv2.waitKey(0)

# Convert image and show its components
Y_channel, Cb_channel, Cr_channel = bgr2ycbcr(rgbsample)
cv2.imshow("Y channel", Y_channel)
cv2.waitKey(0)
cv2.imshow("Cb channel", Cb_channel)
cv2.waitKey(0)
cv2.imshow("Cr channel", Cr_channel)
cv2.waitKey(0)

# Filter and show Y channel
gaussian_filter(Y_channel)
cv2.imshow("Y channel Filtered", Y_channel)
cv2.waitKey(0)

# Filter and show Cb channel
salt_pepper_filter(Cb_channel)
cv2.imshow("Cb channel Filtered", Cb_channel)
cv2.waitKey(0)

# Filter and show Cr channel
frequency_filter(Cr_channel)
cv2.imshow("Cr channel Filtered", Cr_channel)
cv2.waitKey(0)

# Converts channels back to image, shows, and saves
image = ycbcr2bgr(Y_channel, Cb_channel, Cr_channel)
cv2.imshow("Filtered image", image)
cv2.waitKey(0)
cv2.imwrite("Filtered_Image.bmp", image)

cv2.destroyAllWindows()

# ---------- End of main program ----------
