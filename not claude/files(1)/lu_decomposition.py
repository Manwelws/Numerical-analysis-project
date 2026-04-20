import copy
from methods.base import LinearSystem


class LUDecomposition(LinearSystem):
    """
    Solves Ax = B using Doolittle LU decomposition.

    Decomposes A into L (lower, unit diagonal) and U (upper),
    then solves:
        Ly = B  (forward substitution)
        Ux = y  (back substitution)

    Steps record both the decomposition and the substitution stages.
    """

    def _decompose(self, A: list) -> tuple:
        """Returns (L, U) as separate n×n lists."""
        n = len(A)
        L = [[0.0] * n for _ in range(n)]
        U = [[0.0] * n for _ in range(n)]

        for i in range(n):
            L[i][i] = 1.0  # unit diagonal for L

            # ── Fill row i of U ──────────────────────────────────────────
            for j in range(i, n):
                U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))

            # ── Fill column i of L ───────────────────────────────────────
            for j in range(i + 1, n):
                if abs(U[i][i]) < 1e-12:
                    raise ValueError(f"Zero pivot at index {i}. Matrix may be singular.")
                L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]

        return L, U

    def solve(self, A: list, B: list) -> dict:
        self.steps = []
        n = len(A)

        self._record("Original A", copy.deepcopy(A))

        L, U = self._decompose([row[:] for row in A])
        self._record("L matrix", copy.deepcopy(L))
        self._record("U matrix", copy.deepcopy(U))

        # ── Forward substitution: Ly = B ─────────────────────────────────
        y = [0.0] * n
        for i in range(n):
            y[i] = B[i] - sum(L[i][k] * y[k] for k in range(i))
        self._record("Intermediate vector y (from Ly=B)", y[:])

        # ── Back substitution: Ux = y ────────────────────────────────────
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - sum(U[i][k] * x[k] for k in range(i + 1, n))) / U[i][i]
        self._record("Solution vector x", x[:])

        return {"solution": x, "L": L, "U": U, "steps": self.steps}
