from __future__ import annotations

from typing import List, Tuple

IntMatrix = List[List[int]]


class Matrix:

    def __init__(self, matrix: IntMatrix):
        is_valid, invalid_reason = Matrix.is_valid_matrix(matrix)
        if not is_valid:
            raise ValueError(invalid_reason)
        self.matrix: IntMatrix = matrix

    @staticmethod
    def is_valid_matrix(matrix: IntMatrix) -> Tuple[bool, str]:
        """
        return (is_valid, invalid_reason)
        """
        # check if #row == 9
        if len(matrix) != 9:
            return False, f"#row is {len(matrix)}, expect 9."
        # check if #col == 9
        for row_idx, row in enumerate(matrix):
            if len(row) != 9:
                return False, f"{row_idx=} has {len(row)} elements, expect 9."
        # check if any number is in [0, 9)
        for row_idx, row in enumerate(matrix):
            for col_idx, num in enumerate(row):
                if num < 0 or num > 9:
                    return (
                        False,
                        f"matrix[{row_idx}][{col_idx}]={num}, expect in [0, 9].",
                    )
        return True, ""

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        return "\n".join([str(row) + ", " for row in self.matrix])

    def __eq__(self, __value: object) -> bool:
        if self is __value:
            return True
        if not isinstance(__value, Matrix):
            return False
        return self.matrix == __value.matrix
