from abc import ABC, abstractmethod
from copy import deepcopy

from sudoku.matrix import IntMatrix, Matrix


class MatrixLoader(ABC):

    @abstractmethod
    def load(self) -> Matrix:
        pass


class BasicMatrixLoader(MatrixLoader):
    """
    Load a matrix from a 2D List. Make a deep copy.
    """

    def __init__(self, matrix: IntMatrix):
        self.matrix: IntMatrix = matrix

    def load(self) -> Matrix:
        return Matrix(deepcopy(self.matrix))
