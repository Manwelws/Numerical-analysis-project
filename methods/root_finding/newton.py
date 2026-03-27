from methods.base import RootFinding


class Newton(RootFinding):
    def __init__(self, f, f_prime, xi, tol=1e-6, max_iter=100):
        super().__init__(f=f, tol=tol, max_iter=max_iter)
        self.f_ = f_prime
        self.xi = xi

    def solve(self):
        self.iterations = []
        xi = self.xi

        for i in range(1, self.max_iter + 1):
            f_x = self.f(xi)
            f_x_ = self.f_(xi)
            x_new = xi - (f_x / f_x_)

            error = (
                abs((x_new - xi) / x_new) * 100
                if i > 1 and x_new != 0
                else float("inf")
            )
            self._record(iteration=i, x=xi, f_x=f_x, f_x_=f_x_, error=error)

            if error < self.tol:
                return {"root": x_new, "iterations": self.iterations, "converged": True}

            xi = x_new

        return {"root": xi, "iterations": self.iterations, "converged": False}
