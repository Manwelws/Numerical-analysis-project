from methods.base import RootFinding


class Secant(RootFinding):
    def __init__(self, f, xi_1, xi, tol=1e-6, max_iter=100):
        """
        xi_1 : x_{i-1}  (first initial guess)
        xi   : x_i      (second initial guess)
        """
        super().__init__(f=f, tol=tol, max_iter=max_iter)
        self.xi_1 = xi_1
        self.xi = xi

    def solve(self):
        self.iterations = []

        xi_1 = self.xi_1
        xi = self.xi

        for i in range(1, self.max_iter + 1):
            f_xi_1 = self.f(xi_1)
            f_xi = self.f(xi)

            denom = f_xi - f_xi_1
            if denom == 0:
                return {"root": xi, "iterations": self.iterations, "converged": False}

            x_new = xi - f_xi * (xi - xi_1) / denom

            error = (
                abs((x_new - xi) / x_new) * 100
                if x_new != 0
                else float("inf")
            )

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
