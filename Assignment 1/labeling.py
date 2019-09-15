#   Nome: Pedro Victor Rodrigues de Carvalho
#   Matrícula: 17/0113043
#   Universidade de Brasília, 2º semestre de 2018
#   Curso: Engenharia de Computação
#   Matéria: Introdução ao Processamento de Imagens
#   Professor: Alexandre Zaghetto

#   Segunda parte do Assignment 1: Rotulação de componentes conectados em imagem binária
#   Linguagem de programação: Python 3


import cv2
import numpy as np

# --------------------
#   Start of functions


#   The placeNewLabel function simply gives a new value to a position in the label matrix.
def placeNewLabel(matrix, x, y, value):
    matrix[x, y] = value


#       The placeExistingLabel function uses a string its caller supplies as a directive to which neighbour's label
#   should be placed in the current pixel.
def placeExistingLabel(matrix, x, y, string):
    if string == "left":
        matrix[x, y] = matrix[x - 1, y]
    else:
        matrix[x, y] = matrix[x, y - 1]


#       The compareTopLeftLabel function tests if the top and the left neighbours' labels are equal and returns
#   accordingly.
def compareTopLeftLabel(matrix, x, y):
    if matrix[x - 1, y] != matrix[x, y - 1]:
        return 1
    else:
        return 0


#   The establishLabelEquivalence function saves a determined equivalence in an array for future reference.
def establishLabelEquivalence(matrix, x, y, equivalences):
    value1 = matrix[x, y]  # Preference for smaller values of labels.
    value2 = matrix[x, y - 1]
    if value1 < value2:
        equivalences[value2] = value1
    else:
        equivalences[value1] = value2


#      The countsArrayElements function simply counts how many elements in the supplied array are bigger than zero and
#   returns the number of elements found.
def countsArrayElements(array):
    counter = 0
    for i in range(0, array.__len__()):
        if array[i] > 0:
            counter += 1
    return counter


#   The countLabels function used to be in the main program, but since it is used twice it was made to be a function.
def countLabels(img):                                                           # Adds 1px wide border to top and left.
    borderedImage = cv2.copyMakeBorder(img, 1, 0, 1, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    labelMatrix = np.zeros((borderedImage.shape[0], borderedImage.shape[1]), dtype=int)  # Initiates label matrix w/ 0s.
    labelCounter = 1  # Initiates label counter.
    labelEquivalenceArray = [-1] * 501  # Initiates equivalence array with 500 zeroes.

    for y in range(1, borderedImage.shape[1]):          # For every pixel in the image:
        for x in range(1, borderedImage.shape[0]):      # Horizontal scan (whole lines first each time)
            if borderedImage[x, y, 0] == 0:             # If pixel(x, y) is black, then:
                leftpx = borderedImage[x - 1, y, 0]     # Process its top and left neighbours.
                toppx = borderedImage[x, y - 1, 0]

                if (leftpx == 255) and (toppx == 255):  # If both neighbours are white, place new label in pixel(x, y).
                    placeNewLabel(labelMatrix, x, y, labelCounter)
                    labelCounter += 1

                elif (leftpx == 0) != (toppx == 0):     # Else, if only one is black, place its label in pixel(x, y).
                    if leftpx == 0:
                        placeExistingLabel(labelMatrix, x, y, "left")
                    else:
                        placeExistingLabel(labelMatrix, x, y, "top")

                elif leftpx == 0 and toppx == 0:        # Else, if both are black, then:
                    placeExistingLabel(labelMatrix, x, y, "top")  # Label it as one of them (top preference)
                    if compareTopLeftLabel(labelMatrix, x, y):  # If their labels are different, make them be
                        establishLabelEquivalence(labelMatrix, x, y, labelEquivalenceArray)  # equivalent.

#   The number of connected components is equal to the total number of labels minus the number of label equivalences.
    return labelMatrix.max() - countsArrayElements(labelEquivalenceArray)


#   End of functions
# ------------------------
#   Start of main program


#   Connected components labeling
image = cv2.imread("spots.tif")  # Loads image.
print("Total number of labels: ", countLabels(image))  # Prints the number of connected black components in image.

#   Hole labeling
image = cv2.bitwise_not(image)  # Inverts image's binary colors so that holes are black and other components are white.
#   The number of holes is equal to the number of labels -1 because the background is now considered a label.
print("Total number of holes: ", countLabels(image) - 1)  # Prints the number of black connected components in image.


#   End of main program
