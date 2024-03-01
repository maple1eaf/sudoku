from abc import ABC, abstractmethod
from copy import deepcopy
from math import ceil
from typing import List, Tuple, Union

import cv2
import numpy as np
import tensorflow as tf

from sudoku.config import CELL_DIGIT_OCCUPATION_PERCENTAGE
from sudoku.matrix import IntMatrix, Matrix
from sudoku.utils import euclidian_distance, show_image


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

    def __init__(self, img_buffer: bytes, model: tf.keras.Model):
        super().__init__()
        self.img_buffer: bytes = img_buffer
        self.model: tf.keras.Model = (
            model  # tensorflow digit recognization model for 28*28 gray image
        )

        self.original_img: Union[cv2.typing.MatLike, None] = None
        self.gray_img: Union[cv2.UMat, None] = None
        self.blurred_img: Union[cv2.UMat, None] = None
        self.thresholded_img: Union[cv2.UMat, None] = None
        self.sudoku_contour: Union[cv2.UMat, None] = None
        self.ordered_contour_corners: Union[np.array, None] = None
        self.ordered_tf_contour_corners: Union[np.array, None] = None
        self.transformation_matrix: Union[cv2.typing.MatLike, None] = None
        self.sudoku_img: Union[cv2.UMat, None] = None
        self.sudoku_binary_from_original: Union[bytes, None] = None
        self.resized_sudoku_img: Union[cv2.UMat, None] = None

        self._tf_width: Union[float, None] = None
        self._tf_height: Union[float, None] = None
        self._cell_border_mask: Union[np.array, None] = None
        self._predicted_matrix: List[List[int]] = [
            [0 for col in range(9)] for row in range(9)
        ]

    def load(self) -> Matrix:
        # Read the image from memory
        self.original_img = cv2.imdecode(
            np.frombuffer(self.img_buffer, np.uint8), cv2.IMREAD_COLOR
        )
        self.gray_img = cv2.cvtColor(self.original_img, cv2.COLOR_BGR2GRAY)

        # extract sudoku image
        self._decide_contours()
        self._transform_to_aerial_perspective(image=self.thresholded_img)

        self._generate_sudoku_jpg_image_binary()

        # use model to predict cell digits and generate matrix
        self._extract_digits()

        self.loaded_matrix = Matrix(deepcopy(self._predicted_matrix))

        return self.loaded_matrix

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
        self.blurred_img = cv2.GaussianBlur(
            src=self.gray_img, ksize=(21, 21), sigmaX=4, borderType=cv2.BORDER_WRAP
        )

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
        self._tf_width = tf_width
        self._tf_height = tf_height

        self.transformation_matrix = cv2.getPerspectiveTransform(
            src=self.ordered_contour_corners, dst=self.ordered_tf_contour_corners
        )
        # transform the image
        self.sudoku_img = cv2.warpPerspective(
            src=image,
            M=self.transformation_matrix,
            dsize=(int(tf_width), int(tf_height)),
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
                euclidian_distance(top_left, top_right),
                euclidian_distance(buttom_left, buttom_right),
            ]
        )
        max_height = max(
            [
                euclidian_distance(top_left, buttom_left),
                euclidian_distance(top_right, buttom_right),
            ]
        )
        return (max_width, max_height)

    def _extract_digits(self) -> None:
        self.resized_sudoku_img = cv2.resize(
            src=self.sudoku_img, dsize=(270, 540), interpolation=cv2.INTER_AREA
        )

        step_width = self.resized_sudoku_img.shape[1] // 9
        step_height = self.resized_sudoku_img.shape[0] // 9

        # generate a mask to remove cell border, % of the cell width or height is border
        self._cell_border_mask = np.zeros(
            shape=(step_height, step_width), dtype=np.uint8
        )
        SIDE_BORDER_PERCENTAGE = 0.1
        border_width: int = ceil(step_width * SIDE_BORDER_PERCENTAGE)
        border_height: int = ceil(step_height * SIDE_BORDER_PERCENTAGE)
        self._cell_border_mask[
            border_height : (step_height - border_height),
            border_width : (step_width - border_width),
        ] = 255  # 1111 1111

        for row in range(9):
            for col in range(9):
                # (width, height)
                cell_top_left = (step_width * col, step_height * row)
                cell_buttom_right = (step_width * (col + 1), step_height * (row + 1))
                cell: cv2.UMat = self.resized_sudoku_img[
                    cell_top_left[1] : cell_buttom_right[1],
                    cell_top_left[0] : cell_buttom_right[0],
                ]
                pure_cell: Union[cv2.UMat, None] = self._purify_cell(cell=cell)
                if pure_cell is None:
                    continue
                predicted_digit: int = self._predict_digit_in_cell(cell=pure_cell)
                self._predicted_matrix[row][col] = predicted_digit

    def _purify_cell(self, cell: cv2.typing.MatLike) -> Union[cv2.UMat, None]:
        threshed_cell = cv2.threshold(
            src=cell, thresh=127, maxval=255, type=cv2.THRESH_BINARY
        )[1]

        # remove border by using mask
        unborderd_threshed_cell = cv2.bitwise_and(
            src1=threshed_cell, src2=threshed_cell, mask=self._cell_border_mask
        )
        contours, hierarchy = cv2.findContours(
            image=unborderd_threshed_cell.copy(),
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        # if no contour found, the cell is empty
        if len(contours) == 0:
            return None
        digit_contour: cv2.UMat = max(contours, key=cv2.contourArea)
        cell_digit_mask: cv2.UMat = np.zeros(threshed_cell.shape, dtype=np.uint8)
        # thickness=-1 will fill the contour
        cell_digit_mask = cv2.drawContours(
            image=cell_digit_mask,
            contours=[digit_contour],
            contourIdx=0,
            color=255,
            thickness=-1,
        )

        # consider low filled% as noise and ignore as empty cell
        filled_percentage: float = cv2.countNonZero(src=cell_digit_mask) / float(
            cell_digit_mask.shape[0] * cell_digit_mask.shape[1]
        )
        if filled_percentage < CELL_DIGIT_OCCUPATION_PERCENTAGE:
            return None

        pure_cell: cv2.UMat = cv2.bitwise_and(unborderd_threshed_cell, cell_digit_mask)
        return pure_cell

    def _predict_digit_in_cell(self, cell: cv2.UMat) -> int:
        resized_cell = cv2.resize(cell, (28, 28))
        resized_cell = resized_cell.astype(np.float32) / 255.0
        model_input = resized_cell.reshape(-1, 28, 28, 1)
        predicted_digit: int = self.model.predict(model_input, verbose=0).argmax(
            axis=1
        )[0]
        return predicted_digit

    def _generate_sudoku_jpg_image_binary(self) -> None:
        sudoku_img_from_original: cv2.UMat = cv2.warpPerspective(
            src=self.original_img,
            M=self.transformation_matrix,
            dsize=(int(self._tf_width), int(self._tf_height)),
        )
        # convert UMat to binary
        self.sudoku_binary_from_original = cv2.imencode(
            ext=".jpg", img=sudoku_img_from_original
        )[1].tobytes()
