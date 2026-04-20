import numpy as np
from methods.base import LinearSystem


class GaussJordanPivot(LinearSystem):
    def __init__(self, A, B):
        # Enforce float64 data types upon initialization
        self.A = np.array(A, dtype=np.float64)
        self.B = np.array(B, dtype=np.float64).flatten()

    def solve(self):
        self.steps = []
        n = len(self.A)

        # 1. Create the Augmented Matrix cleanly
        M = np.column_stack((self.A, self.B))
        self._record("Initial augmented matrix [A|B]", M.copy().tolist())

        for col in range(n):
            # ── NUMPY MAGIC: Partial Pivoting ──
            # np.argmax finds the index of the maximum value.
            # We slice the column from the current 'col' downwards, find the max absolute value,
            # and add 'col' back to get the true row index.
            max_row = np.argmax(np.abs(M[col:, col])) + col

            if max_row != col:
                # NumPy "Fancy Indexing" allows instant swapping of entire rows in place
                M[[col, max_row]] = M[[max_row, col]]
                self._record(
                    f"Swapped row {col + 1} ↔ row {max_row + 1}", M.copy().tolist()
                )

            pivot = M[col, col]

            # Use 1e-12 threshold for singularity just like your original code
            if abs(pivot) < 1e-12:
                raise ValueError(
                    f"Matrix is singular at column {col + 1}. No unique solution exists."
                )

            # Normalize pivot row
            # We only divide from 'col' onwards to save compute time
            M[col, col:] = M[col, col:] / pivot

            # Eliminate ALL other rows
            for row in range(n):
                if row == col:
                    continue

                factor = M[row, col]
                # Vectorized row subtraction
                M[row, col:] = M[row, col:] - (factor * M[col, col:])

            self._record(f"After eliminating column {col + 1}", M.copy().tolist())

        # Extract the Solution Vector (the last column)
        x = M[:, n]

        self._record("Solution vector x", x.tolist())

        return {"solution": x.tolist(), "steps": self.steps}
