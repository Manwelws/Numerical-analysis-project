from methods.base import RootFinding


class SimpleFixedPoint(RootFinding):
    def __init__(self, g, xi, tol=0.0001, max_iter=50):
        super().__init__(f=g, tol=tol, max_iter=max_iter)
        self.xi = xi

    def solve(self):
        self.iterations = []
        x = self.xi

        for i in range(1, self.max_iter + 1):
            x_new = self.f(x)
            error = abs((x_new - x) / x_new) * 100 if x_new != 0 else float("inf")

            self._record(iteration=i, x=x, g_x=x_new, error=error)

            if error < self.tol:
                return {"root": x_new, "iterations": self.iterations, "converged": True}

            x = x_new

        return {"root": x, "iterations": self.iterations, "converged": False}
