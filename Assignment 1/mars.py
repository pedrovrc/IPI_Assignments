#   Nome: Pedro Victor Rodrigues de Carvalho
#   Matrícula: 17/0113043
#   Universidade de Brasília, 2º semestre de 2018
#   Curso: Engenharia de Computação
#   Matéria: Introdução ao Processamento de Imagens
#   Professor: Alexandre Zaghetto

#   Primeira parte do Assignment 1: Processamento de caminho em imagem topológica
#   Linguagem de programação: Python 3


import cv2
import operator

# --------------------------------
#   Start of functions and classes


#       The Pixel class was created so that the program can handle pixels as variables, rather than avulse numbers.
#       It has two built-in functions: setCoordinates and calcDist2Objective.
#       ->setCoordinates: Receives two numbers that will be stored in the instance to be used as the pixel's position
#   coordinates.
#       ->calcDist2Objective: Also receives coordinates, but instead calculates the distance between the instance's
#   coordinates and the "objective" coordinates it received.
class Pixel:
    def __init__(self):
        self.coordinates = [0, 0]
        self.distance = 0

    def setCoordinates(self, coordi, coordj):
        self.coordinates = [coordi, coordj]

    def calcDist2Objective(self, objectivei, objectivej):
        self.distance = (((self.coordinates[0]-objectivei)**2)+((self.coordinates[1]-objectivej)**2))**0.5


#       The bgr2gray function uses the equation used to calculate the Y value in RGB -> YCbCr conversion to succesfully
#   produce a grayscale version of the original image it received in its arguments.
#       Equation: Y = (R*0.299)+(G*0.587)+(B*0.114)
def bgr2gray(image):
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            image[i, j]=(image[i, j, 2]*0.299)+(image[i, j, 1]*0.587)+(image[i, j, 0]*0.114)


#       The equalize function receives a grayscaled image and creates an array with its histogram of gray levels, then
#   computes the Cumulative Probability Function (CDF) in another array. After that, using the CDF, equalizes the image
#   to maximize the level of detail within 8 bits of gray levels.
def equalizeGrayscale(image):
    sum = 0             # Initializing variables
    histogram = [0] * 256
    CDF = [0] * 256

    for i in range(0, image.shape[0]):      # Creating histogram
        for j in range(0, image.shape[1]):
            histogram[image[i, j, 0]] += 1

    for i in range (0, 256):                # Computing CDF
        sum += ((histogram[i]*3)/image.size)
        CDF[i] = sum

    for i in range(0, image.shape[0]):      # Equalizing image
        for j in range(0, image.shape[1]):
            image[i, j] = CDF[image[i, j, 0]]*255


#       The finds3Closest function computes the surrounding pixels given a center pixel. Then, it removes the previous
#   pixel used so that there is no backtracking. After that, it computes the Euclidean distance from the surrounding
#   pixels to the objective and sorts the array that contains these pixels. Lastly, the 3 pixels with the smaller
#   distance are stored in an array to be used by the function's caller.
def finds3Closest(center, objective, previous, pixList):
    neigh1 = Pixel()            # Define neighbours' variables.
    neigh2 = Pixel()
    neigh3 = Pixel()
    neigh4 = Pixel()
    neigh5 = Pixel()
    neigh6 = Pixel()
    neigh7 = Pixel()
    neigh8 = Pixel()

    neigh1.setCoordinates(center.coordinates[0] - 1, center.coordinates[1] - 1)
    neigh2.setCoordinates(center.coordinates[0] - 1, center.coordinates[1]    )
    neigh3.setCoordinates(center.coordinates[0] - 1, center.coordinates[1] + 1)
    neigh4.setCoordinates(center.coordinates[0],     center.coordinates[1] + 1)
    neigh5.setCoordinates(center.coordinates[0] + 1, center.coordinates[1] + 1)
    neigh6.setCoordinates(center.coordinates[0] + 1, center.coordinates[1]    )
    neigh7.setCoordinates(center.coordinates[0] + 1, center.coordinates[1] - 1)
    neigh8.setCoordinates(center.coordinates[0]    , center.coordinates[1] - 1)

    list = [neigh1, neigh2, neigh3, neigh4, neigh5, neigh6, neigh7, neigh8]     # Arrange them in a list of candidates.

    index2eliminate=-1
    for i in range(0, list.__len__()):              # Eliminate previous pixel from candidates to prevent backtracking.
        if list[i].coordinates == previous.coordinates:
            index2eliminate = i
    if index2eliminate > 0:
        del list[index2eliminate]

    for i in range(0, list.__len__()):              # Calculate distance to objective for each candidate.
        list[i].calcDist2Objective(objective.coordinates[0], objective.coordinates[1])

    list.sort(key=operator.attrgetter('distance'))  # Sort list by distance to objective.

    for i in range(0, 3):                           # The 3 first are the 3 closest to the objective.
        pixList[i].setCoordinates(list[i].coordinates[0], list[i].coordinates[1])


#    The choosesDarkestGray function compares 3 pixels in an array and returns the one with the darkest gray level.
def choosesDarkestGray(image, pixList):
    darkest = pixList[0]
    if image[pixList[1].coordinates[0], pixList[1].coordinates[1], 0] < image[darkest.coordinates[0], darkest.coordinates[1], 0]:
        darkest = pixList[1]
    if image[pixList[2].coordinates[0], pixList[2].coordinates[1], 0] < image[darkest.coordinates[0], darkest.coordinates[1], 0]:
        darkest = pixList[2]
    return darkest


#       The colorsPath function receives two images, a pixel and a level of gray to color the corresponding pixels in
#   these two images.
def colorsPath(original, image, pixel, graylvl):
    original[pixel.coordinates[0], pixel.coordinates[1]] = graylvl
    image[pixel.coordinates[0], pixel.coordinates[1]] = graylvl


#   End of functions and classes
# ------------------------------
#   Start of main program


original = cv2.imread("Mars.bmp")   # Load image.
img = cv2.imread("Mars.bmp")

bgr2gray(img)               # Convert image to grayscale.
equalizeGrayscale(img)      # Equalize the grayscaled image.

originPix = Pixel()         # Define origin and objective pixels.
objectivePix = Pixel()
originPix.setCoordinates(260, 415)
objectivePix.setCoordinates(815, 1000)
originPix.calcDist2Objective(objectivePix.coordinates[0], objectivePix.coordinates[1])

candidate1 = Pixel()        # Define pixel candidates to use in path-searching.
candidate2 = Pixel()
candidate3 = Pixel()
candidates = [candidate1, candidate2, candidate3]   # Create array with them.

previousPix = currentPix = originPix    # Define auxiliary variables.
counter = 0

while currentPix.coordinates != objectivePix.coordinates:   # Loops until the current pixel is the objective.
    colorsPath(original, img, currentPix, 255)              # Colors the current pixel white.
    finds3Closest(currentPix, objectivePix, previousPix, candidates)    # Finds 3 candidates for next move.
    nextPix = choosesDarkestGray(img, candidates)           # Determines the darkest between the 3 candidates.
    previousPix = currentPix
    currentPix = nextPix                                    # Moves the current pixel.
    counter += 1

#   Display the image with the path and save the image. Also save an equalized version of the path for analysis.
cv2.imshow("path", original)
cv2.waitKey(0)
cv2.imwrite("path.bmp", original)
cv2.imwrite("pathEQ.bmp", img)
print("Image file with path saved in program directory.")
print("Actual path length (in pixels):")    # Print the path's length in pixels in the terminal.
print(counter)
print("Straight distance between origin and objective (Euclidean distance):")    # Prints the origin's euclidean
print(originPix.distance)                                               # distance to the objective in the terminal.
cv2.destroyAllWindows()


#   End of main program
