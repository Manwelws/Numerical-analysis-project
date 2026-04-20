import numpy as np
from methods.base import LinearSystem


class GaussElimination(LinearSystem):
    def __init__(self, A: list, B: list) -> dict:
        self.A = np.array(A, dtype=np.float64)
        self.B = np.array(B, dtype=np.float64).flatten()

    def solve(self):
        self.steps = []
        n = len(self.A)
        multipliers = {}

        M = np.column_stack((self.A, self.B))
        self._record("Initial augmented matrix [A|B]", M.copy())

        # forward elimination
        for col in range(n):
            pivot = M[col, col]
            if np.isclose(pivot, 0.0):
                raise ValueError(
                    f"Zero pivot detected at column {col + 1}. System requires partial pivoting."
                )
            for row in range(col + 1, n):
                m = M[row, col] / pivot
                multipliers[(row, col)] = m

                M[row, col:] = M[row, col:] - (m * M[col, col:])

                self._record(f"After eliminating column {col + 1}", M.copy())
        # Back sub
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            x[i] = (M[i, n] - np.dot(M[i, i + 1 : n], x[i + 1 : n])) / M[i, i]

        self._record("Solution vector x", x)
        return {
            "solution": x.tolist(),
            "steps": self.steps,
            "multipliers": multipliers,
            "upper": M.tolist(),
        }
