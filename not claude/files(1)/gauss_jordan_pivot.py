import copy
from methods.base import LinearSystem


class GaussJordanPivot(LinearSystem):
    """
    Solves Ax = B via full row reduction with partial pivoting.
    At each column, swaps the current row with the row that has
    the largest absolute value in that column to improve stability.
    """

    def solve(self, A: list, B: list) -> dict:
        self.steps = []
        n = len(A)

        M = [A[i][:] + [B[i]] for i in range(n)]
        self._record("Initial augmented matrix [A|B]", copy.deepcopy(M))

        for col in range(n):
            # ── Partial Pivot: find row with max |value| in this column ──
            max_row = max(range(col, n), key=lambda r: abs(M[r][col]))
            if max_row != col:
                M[col], M[max_row] = M[max_row], M[col]
                self._record(f"Swapped row {col} ↔ row {max_row}", copy.deepcopy(M))

            pivot = M[col][col]
            if abs(pivot) < 1e-12:
                raise ValueError(f"Matrix is singular at column {col}.")

            # Normalize pivot row
            for j in range(col, n + 1):
                M[col][j] /= pivot

            # Eliminate ALL other rows
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
