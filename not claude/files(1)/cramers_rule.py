import copy
from methods.base import LinearSystem


class CramersRule(LinearSystem):
    """
    Solves Ax = B using Cramer's Rule.

    x_i = det(A_i) / det(A)
    where A_i is A with column i replaced by B.

    NOTE: O(n!) complexity — only practical for small systems (n ≤ 4).
    Raises ValueError for singular matrices (det(A) ≈ 0).
    """

    def _determinant(self, M: list) -> float:
        """Recursive cofactor expansion (fine for small n)."""
        n = len(M)
        if n == 1:
            return M[0][0]
        if n == 2:
            return M[0][0] * M[1][1] - M[0][1] * M[1][0]

        det = 0.0
        for col in range(n):
            minor = [
                [M[row][j] for j in range(n) if j != col]
                for row in range(1, n)
            ]
            det += ((-1) ** col) * M[0][col] * self._determinant(minor)
        return det

    def _replace_column(self, A: list, B: list, col: int) -> list:
        """Returns a copy of A with column `col` replaced by B."""
        n = len(A)
        return [
            [B[row] if j == col else A[row][j] for j in range(n)]
            for row in range(n)
        ]

    def solve(self, A: list, B: list) -> dict:
        self.steps = []
        n = len(A)

        self._record("Original A", copy.deepcopy(A))
        self._record("RHS vector B", B[:])

        det_A = self._determinant(A)
        self._record(f"det(A) = {det_A}", None)

        if abs(det_A) < 1e-12:
            raise ValueError("Matrix is singular (det(A) ≈ 0). Cramer's Rule cannot be applied.")

        x = []
        for i in range(n):
            A_i = self._replace_column(A, B, i)
            det_Ai = self._determinant(A_i)
            xi = det_Ai / det_A
            self._record(f"det(A_{i}) = {det_Ai}  →  x[{i}] = {xi}", copy.deepcopy(A_i))
            x.append(xi)

        self._record("Solution vector x", x[:])
        return {"solution": x, "det_A": det_A, "steps": self.steps}
