import numpy as np
from methods.base import LinearSystem


class CramersRule(LinearSystem):
    def __init__(self, A, B):
        self.A = np.array(A, dtype=np.float64)
        self.B = np.array(B, dtype=np.float64).flatten()

    def solve(self):
        self.steps = []
        n = len(self.A)

        self._record("Original A", self.A.tolist())
        self._record("RHS vector B", self.B.tolist())

        det_A = np.linalg.det(self.A)
        self._record(f"det(A) = {det_A:.6f}", None)

        if np.isclose(det_A, 0.0, atol=1e-12):
            raise ValueError(
                "Matrix is singular (det(A) ≈ 0). Cramer's Rule cannot be applied."
            )

        x = np.zeros(n)

        for i in range(n):
            A_i = self.A.copy()

            A_i[:, i] = self.B

            det_Ai = np.linalg.det(A_i)

            x[i] = det_Ai / det_A

            self._record(
                f"det(A_{i + 1}) = {det_Ai:.6f}  →  x[{i + 1}] = {x[i]:.6f}",
                A_i.tolist(),
            )

        self._record("Solution vector x", x.tolist())

        return {"solution": x.tolist(), "det_A": float(det_A), "steps": self.steps}
