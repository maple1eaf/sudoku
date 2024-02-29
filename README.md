# Sudoku

Features:

- Solve Sudoku
- Load Sudoku from 2D List or image

## Package Usage

### Load a Sudoku puzzle

#### load from integer matrix

```python
import sudoku

matrix: List[List[int]] = ...

puzzle_loader: sudoku.MatrixLoader = sudoku.BasicMatrixLoader(
    matrix=matrix
)
puzzle: sudoku.Matrix = puzzle_loader.load()

# or
puzzle: sudoku.Matrix = sudoku.Matrix(matrix=matrix)
```

#### load from image

```python
import sudoku

img_buffer: bytes = ...
model: tf.keras.Model = ...

loader: sudoku.ImageMatrixLoader = sudoku.ImageMatrixLoader(
    img_buffer=img_buffer, model=model
)
puzzle: sudoku.Matrix = loader.load()
```

### Solve a Sudoku puzzle

```python
import sudoku

puzzle: sudoku.Matrix = ...

solver: sudoku.Solver = sudoku.Solver(puzzle=puzzle)
solver.solve()
```

## Pip

```bash
pip install git+https://github.com/maple1eaf/sudoku.git@main
```
