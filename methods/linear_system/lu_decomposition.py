import copy
from methods.base import LinearSystem
from methods.linear_system.gauss_elimination import GaussElimination


class LUDecomposition(LinearSystem):
    def solve(self, A, B):
        self.steps = []
        n = len(A)
        ge = GaussElimination()
        ge_result = ge.solve(A=[row[:] for row in A], B=B)  # pass args to ge

        for step in ge_result["steps"]:  # saving the history
            self._record(step["description"], step["matrix_state"])

        # get u
        upper_aug = ge_result["upper"]  # [A\B]
        U = [[upper_aug[i][j] for j in range(n)] for i in range(n)]  # [A]
        self._record("U matrix (upper triangular from elimination)", copy.deepcopy(U))

        # get l
        multipliers = ge_result["multipliers"]
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):  # set the diagonal to ones
            L[i][i] = 1.0
        for (row, col), m in multipliers.items():  # loop over the dict keys and values
            L[row][col] = m
        self._record(
            "L matrix (unit lower triangular from multipliers)", copy.deepcopy(L)
        )

        # lc=b
        c = [0.0] * n
        for i in range(n):
            c[i] = B[i] - sum(L[i][k] * c[k] for k in range(i))
        self._record("Intermediate vector y  (from Ly = B)", c[:])

        # ux=c
        x = [0.0] * n
        for i in range(n - 1, -1, -1):  # reverse loop
            x[i] = (c[i] - sum(U[i][k] * x[k] for k in range(i + 1, n))) / U[i][i]
        self._record("Solution vector x  (from Ux = y)", x[:])

        return {"solution": x, "L": L, "U": U, "steps": self.steps}
