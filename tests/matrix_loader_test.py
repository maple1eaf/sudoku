import os
import unittest
from pathlib import Path

import tensorflow as tf
from tests.sudoku_fixture import SudokuFixture

from sudoku.matrix import Matrix
from sudoku.matrix_loader import BasicMatrixLoader, ImageMatrixLoader, MatrixLoader


class TestBasicMatrixLoader(SudokuFixture):

    def test_load(self):
        matrix_loader: MatrixLoader = BasicMatrixLoader(matrix=self.problem_2d_easy)
        matrix: Matrix = matrix_loader.load()
        self.assertNotEquals(id(matrix.matrix), id(self.problem_2d_easy))
        print(matrix)
        print("-------------------")
        print(self.problem_2d_easy)


class TestImageMatrixLoader(unittest.TestCase):

    def setUp(self):
        self.self_folder: Path = Path(__file__).parent.resolve()
        print(self.self_folder)

    def test_load(self):
        img_path: Path = os.path.join(
            self.self_folder, Path("resources/shoot_screen_with_moire.jpg")
        )
        model_path: Path = os.path.join(
            self.self_folder, Path("resources/digit_recognition.h5")
        )
        model: tf.keras.Model = tf.keras.models.load_model(filepath=model_path)
        with open(img_path, "rb") as f:
            loader: ImageMatrixLoader = ImageMatrixLoader(
                img_buffer=f.read(), model=model
            )
            loaded_matrix: Matrix = loader.load()
        print(f"loaded_matrix:\n{loaded_matrix}")
        expected_matrix: Matrix = Matrix(
            matrix=[
                [0, 0, 4, 1, 0, 0, 0, 0, 5],
                [0, 0, 7, 8, 3, 2, 6, 0, 0],
                [3, 9, 0, 7, 0, 0, 8, 0, 0],
                [6, 0, 0, 9, 8, 0, 1, 0, 0],
                [8, 0, 1, 2, 0, 7, 0, 0, 4],
                [0, 4, 9, 0, 1, 3, 0, 0, 2],
                [0, 1, 0, 3, 0, 8, 2, 9, 6],
                [7, 0, 3, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 6, 1, 4, 0, 7],
            ]
        )
        print(f"expected_matrix:\n{expected_matrix}")


if __name__ == "__main__":
    unittest.main()
