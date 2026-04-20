import copy
from methods.base import LinearSystem


class GaussJordan(LinearSystem):
    """
    Solves Ax = B via full row reduction (no partial pivoting).
    Reduces [A|B] to [I|x] directly — no back substitution needed.
    """

    def solve(self, A: list, B: list) -> dict:
        self.steps = []
        n = len(A)

        M = [A[i][:] + [B[i]] for i in range(n)]
        self._record("Initial augmented matrix [A|B]", copy.deepcopy(M))

        for col in range(n):
            pivot = M[col][col]
            if pivot == 0:
                raise ValueError(f"Zero pivot at column {col}. Use partial pivoting.")

            # Normalize pivot row
            for j in range(col, n + 1):
                M[col][j] /= pivot

            # Eliminate ALL other rows (above and below)
            for row in range(n):
                if row == col:
                    continue
                factor = M[row][col]
                for j in range(col, n + 1):
                    M[row][j] -= factor * M[col][j]

            self._record(f"After eliminating column {col}", copy.deepcopy(M))

        x = [M[i][n] for i in range(n)]
        self._record("Solution vector x", x)
        return {"solution": x, "steps": self.steps}
