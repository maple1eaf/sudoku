import os
import unittest
from pathlib import Path

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
        with open(img_path, "rb") as f:
            loader: ImageMatrixLoader = ImageMatrixLoader(img_buffer=f.read())
            loader.load()


if __name__ == "__main__":
    unittest.main()
