import unittest

from tests.sudoku_fixture import SudokuFixture

from sudoku.matrix import Matrix
from sudoku.matrix_loader import BasicMatrixLoader, MatrixLoader


class TestBasicMatrixLoader(SudokuFixture):

    def test_load(self):
        matrix_loader: MatrixLoader = BasicMatrixLoader(matrix=self.problem_2d_easy)
        matrix: Matrix = matrix_loader.load()
        self.assertNotEquals(id(matrix.matrix), id(self.problem_2d_easy))
        print(matrix)
        print("-------------------")
        print(self.problem_2d_easy)


if __name__ == "__main__":
    unittest.main()
