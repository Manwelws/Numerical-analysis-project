import copy
from methods.base import LinearSystem


class GaussElimination(LinearSystem):
    def solve(self, A, B):
        self.steps = []

        M = [A[i][:] + B[i] for i in range(len(A))]
        self._record("Initial augmented matrix [A|B]", copy.deepcopy(M))
        n = len(A)
        multipliers = {}  # for lu

        # forward elimination
        for col in range(n):
            pivot = M[col][col]
            if pivot == 0:
                raise ValueError(f"Zero pivot at column {col}. Use partial pivoting.")

            for row in range(col + 1, n):
                m = M[row][col] / pivot
                multipliers[(row, col)] = m

                for j in range(col, n + 1):
                    M[row][j] -= m * M[col][j]

            self._record(f"After eliminating column {col}", copy.deepcopy(M))

        # Back sub
        x = [0.0] * n  # creates a row vector of length n to save right answers
        for i in range(n - 1, -1, -1):  # reverse loop
            x[i] = M[i][n]  # loop through the 4th colm
            for j in range(i + 1, n):
                x[i] -= M[i][j] * x[j]
            x[i] /= M[i][i]

        self._record("Solution vector x", x)
        return {
            "solution": x,
            "steps": self.steps,
            "multipliers": multipliers,
            "upper": M,
        }
