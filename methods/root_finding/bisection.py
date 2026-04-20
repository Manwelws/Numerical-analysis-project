from methods.base import RootFinding


class Bisection(RootFinding):
    def __init__(self, f, xl, xu, tol=0.0001, max_iter=50):
        super().__init__(f, tol, max_iter)
        self.xl = xl
        self.xu = xu

    def solve(self):
        self.iterations = []

        xl, xu = self.xl, self.xu

        if self.f(xl) * self.f(xu) > 0:
            raise ValueError("Invalid interval: f(xl) and f(xu) have the same sign.")

        xr = None
        xr_old = None

        for i in range(1, self.max_iter + 1):
            xr_old = xr
            xr = (xl + xu) / 2

            f_xl = self.f(xl)
            f_xu = self.f(xu)
            f_xr = self.f(xr)

            error = (
                abs((xr - xr_old) / xr) * 100
                if xr_old is not None and xr != 0
                else float("inf")
            )

            self._record(
                iteration=i,
                xl=xl,
                f_xl=f_xl,
                xu=xu,
                f_xu=f_xu,
                xr=xr,
                f_xr=f_xr,
                error=error,
            )

            if error < self.tol:
                return {"root": xr, "iterations": self.iterations, "converged": True}

            if f_xl * f_xr < 0:
                xu = xr
            else:
                xl = xr

        return {"root": xr, "iterations": self.iterations, "converged": False}
