import sudoku
from sudoku.config import CUSTOM_PROBLEM


def main():
    problem_loader: sudoku.MatrixLoader = sudoku.BasicMatrixLoader(
        matrix=CUSTOM_PROBLEM
    )
    problem: sudoku.Matrix = problem_loader.load()
    solver: sudoku.Solver = sudoku.Solver(problem=problem)
    solver.solve()


if __name__ == "__main__":
    main()
