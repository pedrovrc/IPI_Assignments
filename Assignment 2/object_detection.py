import cv2
import numpy as np
import math


def equalizeGrayscale(image):
    sum = 0             # Initializing variables
    histogram = [0] * 256
    CDF = [0] * 256

    for i in range(0, image.shape[0]):      # Creating histogram
        for j in range(0, image.shape[1]):
            histogram[image[i, j]] += 1

    for i in range (0, 256):                # Computing CDF
        sum += ((histogram[i])/image.size)
        CDF[i] = sum

    for i in range(0, image.shape[0]):      # Equalizing image
        for j in range(0, image.shape[1]):
            image[i, j] = CDF[image[i, j]]*255


def crop_image(image, x0, y0, x, y):
    crop = np.zeros((x - x0, y - y0), dtype=np.uint8)
    crop[:, :] = image[x0:x, y0:y]
    return crop


def draw_rectangle(image, side_dim_x, side_dim_y, x, y, line_width, color):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i >= x and i <= x + line_width or j >= y and j <= y + line_width:
                if i <= x + side_dim_x or j <= y + side_dim_y:
                    image[i, j] = color


def mean(image):
    return 0


def convolution_sum(matrix1, matrix2):
    pass


def cross_correlation(image, temp):
    pass


def template_matching(image, temp, threshold):
    img_h, img_w = image.shape
    temp_h, temp_w = temp.shape
    correl_matrix = np.zeros((img_h - temp_h, img_w - temp_w), dtype=np.float)

    for i in range(0, correl_matrix.shape[0]):
        for j in range(0, correl_matrix.shape[1]):
            region = crop_image(image, i, j, i + temp_h, j + temp_w)
            #correl_matrix[i, j] = cross_correlation(region, temp)

    correl_matrix[288, 288] = 0.9

    for i in range(0, correl_matrix.shape[0]):
        for j in range(0, correl_matrix.shape[1]):
            if correl_matrix[i, j] >= threshold:
                draw_rectangle(image, temp_w, temp_h, j, i, 3, 0)


image = cv2.imread('Filtered_Image.bmp')
cv2.imshow('Reference image', image)
cv2.waitKey(0)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalizeGrayscale(image)

cv2.imshow('Equalized gray scale reference', image)
cv2.waitKey(0)

# Define template for searching
template = crop_image(image, 485, 402, 610, 498)
cv2.imshow('Template', template)
cv2.waitKey(0)

# Scan whole image to find best matches based on specified threshold values
template_matching(image, template, 0.8)

cv2.imshow('Objects found', image)
cv2.waitKey(0)

cv2.destroyAllWindows()