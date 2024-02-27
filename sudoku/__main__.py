import sudoku
from sudoku.config import CUSTOM_PROBLEM


def main():
    problem: sudoku.Matrix = sudoku.Matrix(matrix=CUSTOM_PROBLEM)
    solver: sudoku.Solver = sudoku.Solver(problem=problem)
    solver.solve()


if __name__ == "__main__":
    main()
