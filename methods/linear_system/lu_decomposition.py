import numpy as np
from methods.base import LinearSystem
from methods.linear_system.gauss_elimination import GaussElimination


class LUDecomposition(LinearSystem):
    def __init__(self, A, B):
        self.A = np.array(A, dtype=np.float64)
        self.B = np.array(B, dtype=np.float64).flatten()

    def solve(self):
        self.steps = []
        n = len(self.A)
        ge = GaussElimination(A=self.A, B=self.B)
        ge_result = ge.solve()

        for step in ge_result.get("steps", []):
            self._record(step["description"], step["matrix_state"])

        # get u
        M = np.array(ge_result["upper"])
        U = M[:, :n]
        self._record("U matrix (upper triangular from elimination)", U.tolist())

        # get l
        L = np.eye(n)
        multipliers = ge_result["multipliers"]

        for (row, col), m in multipliers.items():
            L[row, col] = m
        self._record("L matrix (unit lower triangular from multipliers)", L.tolist())

        # lc=b
        c = np.zeros(n)
        for i in range(n):
            c[i] = self.B[i] - np.dot(L[i, :i], c[:i])
        self._record("Intermediate vector y (from Ly = B)", c.tolist())

        # ux=c
        x = np.zeros(n)
        for i in range(n - 1, -1, -1):
            # Using np.dot() again, exactly like in Gauss Elimination
            x[i] = (c[i] - np.dot(U[i, i + 1 :], x[i + 1 :])) / U[i, i]

        self._record("Solution vector x (from Ux = y)", x.tolist())

        return {
            "solution": x.tolist(),
            "L": L.tolist(),
            "U": U.tolist(),
            "steps": self.steps,
        }
