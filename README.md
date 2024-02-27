# Sudoku

Features:

- Solve Sudoku
- Load Sudoku from 2D List or image

## Package Usage

### Solve a Sudoku

```python
import sudoku

matrix: List[List[int]] = ...

problem: sudoku.Matrix = sudoku.Matrix(matrix=matrix)
solver: sudoku.Solver = sudoku.Solver(problem=problem)
solver.solve()

# or from basic loader
problem_loader: sudoku.MatrixLoader = sudoku.BasicMatrixLoader(
    matrix=matrix
)
problem: sudoku.Matrix = problem_loader.load()
solver: sudoku.Solver = sudoku.Solver(problem=problem)
solver.solve()
```

## Pip

```bash
pip install git+https://github.com/maple1eaf/sudoku.git@main
```
