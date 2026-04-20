import numpy as np
from methods.base import LinearSystem


class GaussJordan(LinearSystem):
    def __init__(self, A, B):
        self.A = np.array(A, dtype=np.float64)
        self.B = np.array(B, dtype=np.float64).flatten()

    def solve(self) -> dict:
        self.steps = []
        n = len(self.A)

        M = np.column_stack((self.A, self.B))
        self._record("Initial augmented matrix [A|B]", M.copy().tolist())

        for col in range(n):
            pivot = M[col, col]

            # Safe floating-point zero check
            if np.isclose(pivot, 0.0):
                raise ValueError(
                    f"Zero pivot detected at column {col + 1}. System requires partial pivoting."
                )

            # THE NUMPY MAGIC: Normalize the entire pivot row instantly
            # We only divide from 'col' onwards since everything before it is already 0
            M[col, col:] = M[col, col:] / pivot

            # Eliminate ALL other rows (above and below)
            for row in range(n):
                if row == col:
                    continue

                factor = M[row, col]
                # Vectorized row subtraction
                M[row, col:] = M[row, col:] - (factor * M[col, col:])

            self._record(f"After eliminating column {col + 1}", M.copy().tolist())

        x = M[:, n]

        self._record("Solution vector x", x.tolist())

        return {"solution": x.tolist(), "steps": self.steps}
