from methods.base import RootFinding


class Secant(RootFinding):
    def __init__(self, f, xi, xi_1, tol=1e-6, max_iter=100):
        super().__init__(f=f, tol=tol, max_iter=max_iter)
        self.xi = xi
        self.xi_1 = xi_1

    def solve(self):
        self.iterations = []
        xi = self.xi
        xi_1 = self.xi_1

        for i in range(1, self.max_iter + 1):
            f_xi = self.f(xi)
            f_xi_1 = self.f(xi_1)

            if f_xi_1 - f_xi == 0:
                return {"root": xi, "iterations": self.iterations, "converged": False}

            x_new = xi - (f_xi * (xi_1 - xi)) / (f_xi_1 - f_xi)
            error = abs((x_new - xi) / x_new) * 100 if x_new != 0 else float("inf")

            self._record(
                iteration=i,
                xi_1=xi_1,
                f_xi_1=f_xi_1,
                xi=xi,
                f_xi=f_xi,
                x_new=x_new,
                error=error,
            )

            if error < self.tol:
                return {"root": x_new, "iterations": self.iterations, "converged": True}

            xi_1 = xi
            xi = x_new

        return {"root": xi, "iterations": self.iterations, "converged": False}
