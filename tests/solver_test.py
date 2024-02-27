import unittest
from copy import deepcopy
from typing import List

from tests.sudoku_fixture import SudokuFixture

from sudoku.matrix import Matrix
from sudoku.solver import Solver


class TestSolver(SudokuFixture):

    def setUp(self):
        super().setUp()
        self.problem_easy: Matrix = Matrix(deepcopy(self.problem_2d_easy))
        self.answer_easy: Matrix = Matrix(deepcopy(self.answer_2d_easy))
        self.problem_hard: Matrix = Matrix(deepcopy(self.problem_2d_hard))
        self.answer_hard: Matrix = Matrix(deepcopy(self.answer_2d_hard))
        self.problem_ash: Matrix = Matrix(deepcopy(self.problem_2d_ash))
        self.answer_ash: Matrix = Matrix(deepcopy(self.answer_2d_ash))
        self.problem_wrong: Matrix = Matrix(deepcopy(self.problem_2d_wrong))

    def test_solve_easy(self):
        solver: Solver = Solver(problem=self.problem_easy)
        self.assertTrue(solver.solve())
        self.assertEqual(solver.problem, self.problem_easy)
        self.assertEqual(solver.answer, self.answer_easy)

    def test_solve_hard(self):
        solver: Solver = Solver(problem=self.problem_hard)
        self.assertTrue(solver.solve())
        self.assertEqual(solver.problem, self.problem_hard)
        self.assertEqual(solver.answer, self.answer_hard)

    def test_solve_ash(self):
        solver: Solver = Solver(problem=self.problem_ash)
        self.assertTrue(solver.solve())
        self.assertEqual(solver.problem, self.problem_ash)
        self.assertEqual(solver.answer, self.answer_ash)

    def test_solve_wrong(self):
        solver: Solver = Solver(problem=self.problem_wrong)
        self.assertFalse(solver.solve())


if __name__ == "__main__":
    unittest.main()
