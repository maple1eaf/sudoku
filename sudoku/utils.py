import numpy as np
import cv2

def euclidian_distance(point1: np.array, point2: np.array):
    # Calcuates the euclidian distance between the point1 and point2
    return np.linalg.norm(point1 - point2)

def show_image(image):
    cv2.namedWindow("Image")
    cv2.moveWindow("Image", 40, 30)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()