import copy
from methods.base import LinearSystem


class GaussJordan(LinearSystem):
    def solve(self, A, B):
        self.steps = []
        n = len(A)

        M = [A[i][:] + [B[i]] for i in range(n)]
        self._record("Initial augmented matrix [A|B]", copy.deepcopy(M))

        for col in range(n):
            pivot = M[col][col]
            if pivot == 0:
                raise ValueError(f"Zero pivot at column {col}. Use partial pivoting.")
