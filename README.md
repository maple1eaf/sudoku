# Sudoku

Features:

- Solve Sudoku
- Load Sudoku from 2D List or image

## Package Usage

### Solve a Sudoku

```python
import sudoku

problem: sudoku.Matrix = sudoku.Matrix(matrix=CUSTOM_PROBLEM)
solver: sudoku.Solver = sudoku.Solver(problem=problem)
solver.solve()
```

## Pip

```bash
pip install git+https://github.com/maple1eaf/sudoku.git@main
```
