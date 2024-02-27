import unittest

from tests.sudoku_fixture import SudokuFixture

from sudoku.matrix import Matrix


class TestMatrix(SudokuFixture):

    def test_repr(self):
        matrix: Matrix = Matrix(self.problem_2d_easy)
        print(matrix)
        self.assertEqual(
            repr(matrix),
            "[3, 7, 0, 0, 1, 9, 0, 4, 8], \n"
            + "[0, 0, 8, 0, 4, 0, 0, 2, 0], \n"
            + "[0, 0, 5, 0, 8, 0, 0, 0, 0], \n"
            + "[0, 0, 6, 2, 0, 0, 0, 7, 9], \n"
            + "[0, 0, 9, 1, 3, 7, 8, 0, 0], \n"
            + "[5, 3, 0, 0, 0, 8, 2, 0, 0], \n"
            + "[0, 0, 0, 0, 7, 0, 6, 0, 0], \n"
            + "[0, 8, 0, 0, 0, 0, 9, 0, 0], \n"
            + "[2, 6, 0, 3, 9, 0, 0, 8, 7], ",
        )


if __name__ == "__main__":
    unittest.main()
