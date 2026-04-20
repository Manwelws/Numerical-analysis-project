import copy
from methods.base import LinearSystem


class GaussElimination(LinearSystem):
    """
    Solves Ax = B via forward elimination + back substitution.
    A  : n×n list of lists  (coefficients)
    B  : n×1 list           (RHS)
    """

    def solve(self, A: list, B: list) -> dict:
        self.steps = []
        n = len(A)

        # Build augmented matrix [A | B]
        M = [A[i][:] + [B[i]] for i in range(n)]
        self._record("Initial augmented matrix [A|B]", copy.deepcopy(M))

        # ── Forward Elimination ──────────────────────────────────────────
        for col in range(n):
            pivot = M[col][col]
            if pivot == 0:
                raise ValueError(f"Zero pivot at column {col}. Use partial pivoting.")

            for row in range(col + 1, n):
                factor = M[row][col] / pivot
                for j in range(col, n + 1):
                    M[row][j] -= factor * M[col][j]

            self._record(f"After eliminating column {col}", copy.deepcopy(M))

        # ── Back Substitution ────────────────────────────────────────────
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = M[i][n]
            for j in range(i + 1, n):
                x[i] -= M[i][j] * x[j]
            x[i] /= M[i][i]

        self._record("Solution vector x", x)
        return {"solution": x, "steps": self.steps}
