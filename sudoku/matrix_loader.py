from abc import ABC, abstractmethod
from copy import deepcopy
from typing import List, Tuple, Union

import cv2
import numpy as np

from sudoku.matrix import IntMatrix, Matrix


class MatrixLoader(ABC):

    def __init__(self):
        self.loaded_matrix: Union[Matrix, None] = None

    @abstractmethod
    def load(self) -> Matrix:
        pass


class BasicMatrixLoader(MatrixLoader):
    """
    Load a matrix from a 2D List. Make a deep copy.
    """

    def __init__(self, matrix: IntMatrix):
        super().__init__()
        self.matrix: IntMatrix = matrix

    def load(self) -> Matrix:
        self.loaded_matrix = Matrix(deepcopy(self.matrix))
        return self.loaded_matrix


class ImageMatrixLoader(MatrixLoader):
    """
    Load a matrix from an image.
    """

    def __init__(self, img_buffer):
        self.img_buffer = img_buffer

        self.original_img: Union[cv2.typing.MatLike, None] = None
        self.gray_img: Union[cv2.UMat, None] = None
        self.blurred_img: Union[cv2.UMat, None] = None
        self.thresholded_img: Union[cv2.UMat, None] = None
        self.sudoku_contour: Union[cv2.UMat, None] = None
        self.ordered_contour_corners: Union[np.array, None] = None
        self.ordered_tf_contour_corners: Union[np.array, None] = None
        self.sudoku_img: Union[cv2.UMat, None] = None
        self.resized_sudoku_img: Union[cv2.UMat, None] = None

    def load(self) -> Matrix:
        # Read the image from memory
        self.original_img = cv2.imdecode(
            np.frombuffer(self.img_buffer, np.uint8), cv2.IMREAD_COLOR
        )
        self.gray_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)

        # extract sudoku image
        self._decide_contours()
        self._transform_to_aerial_perspective(image=self.original_img)

        self.resized_sudoku_img = cv2.resize(
            src=self.sudoku_img, dsize=(250, 250), interpolation=cv2.INTER_AREA
        )

        # ImageMatrixLoader.show_image(self.resized_sudoku_img)

    def _decide_contours(self) -> None:
        """
        from the gray image, get the sudoku's contour which consists of 4 corners.

        steps:
        1. get blurred img.
        2. get a binary image. object should be white, background should be black.
        3. find all objects' contours.
        4. sort contours by the area. the biggest contour should refer to sudoku.
        5. get a approximated sudoku contour with 4 points referring to the 4 corners of the sudoku.
        """
        self.blurred_img = cv2.GaussianBlur(self.gray_img, (7, 7), 3)
        # self.blurred_img = cv2.medianBlur(self.gray_img, 3)

        # https://docs.opencv.org/4.9.0/d7/d4d/tutorial_py_thresholding.html
        self.thresholded_img = cv2.adaptiveThreshold(
            src=self.blurred_img,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=11,
            C=3,
        )

        # https://docs.opencv.org/4.9.0/d4/d73/tutorial_py_contours_begin.html
        contours, hierarchy = cv2.findContours(
            image=self.thresholded_img,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        sudoku_contour: cv2.UMat = max(contours, key=cv2.contourArea)
        sudoku_contour_perimeter: float = cv2.arcLength(
            curve=sudoku_contour, closed=True
        )
        approx_sudoku_contour: cv2.UMat = cv2.approxPolyDP(
            curve=sudoku_contour, epsilon=0.02 * sudoku_contour_perimeter, closed=True
        )
        if approx_sudoku_contour.shape[0] != 4:
            raise ValueError(
                f"Expect get an approximated contour consists of 4 corners, but get {approx_sudoku_contour.shape[0]} points. We may need to adjust the `epsilon` value."
            )
        self.sudoku_contour = approx_sudoku_contour

    def _transform_to_aerial_perspective(self, image) -> None:
        self._organize_corners()
        tf_width, tf_height = self._determine_transformed_size()
        self.ordered_tf_contour_corners = np.array(
            [
                [0, 0],
                [0, tf_height - 1],
                [tf_width - 1, tf_height - 1],
                [tf_width - 1, 0],
            ],
            dtype=np.float32,
        )

        transformation_matrix: cv2.typing.MatLike = cv2.getPerspectiveTransform(
            src=self.ordered_contour_corners, dst=self.ordered_tf_contour_corners
        )
        # transform the image
        self.sudoku_img = cv2.warpPerspective(
            src=image, M=transformation_matrix, dsize=(int(tf_width), int(tf_height))
        )

    def _organize_corners(self) -> None:
        # organize corners as [top_left, buttom_left, buttom_right, top_right]
        corners: np.array = np.reshape(self.sudoku_contour, (4, 2))
        # (width, height)
        centroid: np.array = sum(corners) / 4
        for corner in corners:
            vector: np.array = corner - centroid
            if vector[0] < 0 and vector[1] < 0:
                top_left = corner
            elif vector[0] < 0 and vector[1] > 0:
                buttom_left = corner
            elif vector[0] > 0 and vector[1] < 0:
                top_right = corner
            elif vector[0] > 0 and vector[1] > 0:
                buttom_right = corner
            else:
                raise ValueError(f"Impossible shape for corners: {corners}")
        self.ordered_contour_corners = np.array(
            [top_left, buttom_left, buttom_right, top_right], dtype="float32"
        )

    def _determine_transformed_size(self) -> Tuple[float, float]:
        """
        use max width and height as the transformed_size.

        return as (width, height)
        """
        top_left, buttom_left, buttom_right, top_right = self.ordered_contour_corners
        max_width = max(
            [
                ImageMatrixLoader.euclidian_distance(top_left, top_right),
                ImageMatrixLoader.euclidian_distance(buttom_left, buttom_right),
            ]
        )
        max_height = max(
            [
                ImageMatrixLoader.euclidian_distance(top_left, buttom_left),
                ImageMatrixLoader.euclidian_distance(top_right, buttom_right),
            ]
        )
        return (max_width, max_height)

    @staticmethod
    def euclidian_distance(point1: np.array, point2: np.array):
        # Calcuates the euclidian distance between the point1 and point2
        return np.linalg.norm(point1 - point2)

    @staticmethod
    def show_image(image):
        cv2.namedWindow("Image")
        cv2.moveWindow("Image", 40, 30)
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
