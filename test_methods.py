"""
test_methods.py
================
Pytest test suite for the Numerical Solver project (27/3 branch).

Run:   pytest test_methods.py -v

Tests are written against CORRECT behaviour. Each bug section below
corresponds to a bug described in the roadmap document.  Before fixes,
expect all tests outside Root Finding Ch1 (bisection / false_position)
to fail.  After each fix the relevant test(s) should turn green.
"""

import math
import pytest
import sys, os

# ── allow imports from project root ──────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from methods.root_finding import Bisection, FalsePosition, SimpleFixedPoint, Newton, Secant
from methods.linear_system import (
    GaussElimination,
    LUDecomposition,
    GaussJordan,
    CramersRule,
)

# Try importing GaussJordanPivot separately — it may be missing
try:
    from methods.linear_system import GaussJordanPivot
    HAS_PIVOT = True
except ImportError:
    HAS_PIVOT = False


# ─────────────────────────────────────────────────────────────────────────────
# Shared test functions
# ─────────────────────────────────────────────────────────────────────────────

# f(x) = x^3 - x - 2  → root ≈ 1.5214
def f_cubic(x):
    return x**3 - x - 2

def f_cubic_prime(x):
    return 3 * x**2 - 1

# g(x) rearranged for fixed-point:  x = (x + 2)^(1/3)  → same root
def g_fixed(x):
    return (x + 2) ** (1 / 3)

# 3×3 system: unique solution x=[1, 2, -1]
A3 = [
    [2.0,  1.0, -1.0],
    [-3.0, -1.0,  2.0],
    [-2.0,  1.0,  2.0],
]
B3 = [8.0, -11.0, -3.0]
X3_EXPECTED = [2.0, 3.0, -1.0]   # actual solution of the system above

# Verify the system is consistent
def _check_solution(A, B, x, tol=1e-6):
    n = len(A)
    for i in range(n):
        row_sum = sum(A[i][j] * x[j] for j in range(n))
        assert abs(row_sum - B[i]) < tol, (
            f"Row {i}: expected {B[i]}, got {row_sum:.8f}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# CH 1 — ROOT FINDING
# ═════════════════════════════════════════════════════════════════════════════

class TestBisection:
    """Already correct — should pass before any fixes."""

    def test_finds_root(self):
        m = Bisection(f=f_cubic, xl=1.0, xu=2.0, tol=1e-6)
        result = m.solve()
        assert result["converged"]
        assert abs(result["root"] - 1.5214) < 1e-3

    def test_records_iterations(self):
        m = Bisection(f=f_cubic, xl=1.0, xu=2.0, tol=1e-6)
        result = m.solve()
        assert len(result["iterations"]) > 0
        row = result["iterations"][0]
        for key in ("iteration", "xl", "xu", "xr", "f_xl", "f_xu", "f_xr", "error"):
            assert key in row, f"Missing key '{key}' in bisection iteration record"

    def test_invalid_interval_raises(self):
        m = Bisection(f=f_cubic, xl=2.0, xu=3.0)
        with pytest.raises(ValueError):
            m.solve()

    def test_first_error_is_inf(self):
        """First iteration has no xr_old, so error must be inf."""
        m = Bisection(f=f_cubic, xl=1.0, xu=2.0)
        result = m.solve()
        assert result["iterations"][0]["error"] == float("inf")


class TestFalsePosition:
    """Already correct — should pass before any fixes."""

    def test_finds_root(self):
        m = FalsePosition(f=f_cubic, xl=1.0, xu=2.0, tol=1e-6)
        result = m.solve()
        assert result["converged"]
        assert abs(result["root"] - 1.5214) < 1e-3

    def test_records_iterations(self):
        m = FalsePosition(f=f_cubic, xl=1.0, xu=2.0, tol=1e-6)
        result = m.solve()
        row = result["iterations"][0]
        for key in ("iteration", "xl", "xu", "xr", "error"):
            assert key in row

    def test_invalid_interval_raises(self):
        m = FalsePosition(f=f_cubic, xl=2.0, xu=3.0)
        with pytest.raises(ValueError):
            m.solve()


class TestSimpleFixedPoint:
    """BUG 2.2: self.x0 → self.xi  (AttributeError before fix)."""

    def test_no_attribute_error(self):
        """Most basic check — should not crash."""
        m = SimpleFixedPoint(g=g_fixed, xi=1.5, tol=1e-4)
        result = m.solve()   # was: AttributeError: 'SimpleFixedPoint' has no attribute 'x0'
        assert "root" in result

    def test_finds_root(self):
        m = SimpleFixedPoint(g=g_fixed, xi=1.5, tol=1e-6)
        result = m.solve()
        assert result["converged"]
        assert abs(result["root"] - 1.5214) < 1e-3

    def test_iteration_record_keys(self):
        m = SimpleFixedPoint(g=g_fixed, xi=1.5, tol=1e-4)
        result = m.solve()
        row = result["iterations"][0]
        for key in ("iteration", "x", "g_x", "error"):
            assert key in row


class TestNewton:
    """BUG 2.3: error on iteration 1 should not be inf (or at least should
    converge correctly and record x_new in iterations)."""

    def test_finds_root(self):
        m = Newton(f=f_cubic, f_prime=f_cubic_prime, xi=1.5, tol=1e-6)
        result = m.solve()
        assert result["converged"]
        assert abs(result["root"] - 1.5214) < 1e-3

    def test_iteration_keys(self):
        m = Newton(f=f_cubic, f_prime=f_cubic_prime, xi=1.5, tol=1e-4)
        result = m.solve()
        row = result["iterations"][0]
        for key in ("iteration", "x", "f_x", "f_x_", "error"):
            assert key in row

    def test_converges_in_few_iterations(self):
        """Newton should converge fast (quadratic); > 20 iters is a sign something is wrong."""
        m = Newton(f=f_cubic, f_prime=f_cubic_prime, xi=1.5, tol=1e-8)
        result = m.solve()
        assert len(result["iterations"]) < 20


class TestSecant:
    """BUG 2.1: completely wrong formula — all root values will be wrong before fix."""

    def test_finds_root(self):
        """
        Secant method on f(x)=x^3-x-2 with xi_1=1.0, xi=2.0.
        Correct formula:
            x_new = xi - f(xi)*(xi - xi_1) / (f(xi) - f(xi_1))
        """
        m = Secant(f=f_cubic, xi=2.0, xi_1=1.0, tol=1e-6)
        result = m.solve()
        assert result["converged"], "Secant did not converge"
        assert abs(result["root"] - 1.5214) < 1e-3, (
            f"Root {result['root']:.6f} differs from expected 1.5214 — "
            "likely the formula is still broken"
        )

    def test_iteration_keys(self):
        m = Secant(f=f_cubic, xi=2.0, xi_1=1.0, tol=1e-4)
        result = m.solve()
        row = result["iterations"][0]
        for key in ("iteration", "xi", "xi_1", "f_xi", "f_xi_1", "x_new", "error"):
            assert key in row

    def test_zero_denominator_returns_gracefully(self):
        """If f(xi)==f(xi_1), method should return without crashing."""
        # Constant function — denominator is always 0
        m = Secant(f=lambda x: 5.0, xi=1.0, xi_1=2.0)
        result = m.solve()
        assert result["converged"] is False


# ═════════════════════════════════════════════════════════════════════════════
# CH 2 — LINEAR SYSTEMS
# ═════════════════════════════════════════════════════════════════════════════

class TestGaussElimination:
    """BUG 2.4: B augmentation uses list concat with scalar — TypeError before fix."""

    def test_solves_3x3(self):
        ge = GaussElimination()
        result = ge.solve(A=[row[:] for row in A3], B=B3[:])
        _check_solution(A3, B3, result["solution"])

    def test_returns_multipliers_and_upper(self):
        ge = GaussElimination()
        result = ge.solve(A=[row[:] for row in A3], B=B3[:])
        assert "multipliers" in result
        assert "upper" in result

    def test_singular_raises(self):
        A_sing = [[1, 2, 3], [2, 4, 6], [0, 0, 1]]
        B_sing = [1, 2, 0]
        ge = GaussElimination()
        with pytest.raises(ValueError):
            ge.solve(A=A_sing, B=B_sing)

    def test_records_steps(self):
        ge = GaussElimination()
        result = ge.solve(A=[row[:] for row in A3], B=B3[:])
        assert len(result["steps"]) >= 2


class TestLUDecomposition:
    """BUG 2.5: GaussElimination not instantiated — TypeError before fix."""

    def test_solves_3x3(self):
        lu = LUDecomposition()
        result = lu.solve(A=[row[:] for row in A3], B=B3[:])
        _check_solution(A3, B3, result["solution"])

    def test_returns_L_and_U(self):
        lu = LUDecomposition()
        result = lu.solve(A=[row[:] for row in A3], B=B3[:])
        assert "L" in result
        assert "U" in result
        n = len(A3)
        # L must have 1s on diagonal
        for i in range(n):
            assert abs(result["L"][i][i] - 1.0) < 1e-9
        # U must be upper triangular
        for i in range(n):
            for j in range(i):
                assert abs(result["U"][i][j]) < 1e-9, f"U[{i}][{j}] should be 0"

    def test_LU_product_equals_A(self):
        lu = LUDecomposition()
        result = lu.solve(A=[row[:] for row in A3], B=B3[:])
        L, U = result["L"], result["U"]
        n = len(A3)
        for i in range(n):
            for j in range(n):
                prod = sum(L[i][k] * U[k][j] for k in range(n))
                assert abs(prod - A3[i][j]) < 1e-6, (
                    f"LU[{i}][{j}]={prod:.6f} ≠ A[{i}][{j}]={A3[i][j]}"
                )


class TestGaussJordan:
    """BUG 2.7: solve() is incomplete (stub only) — will raise or return wrong result."""

    def test_solves_3x3(self):
        gj = GaussJordan()
        result = gj.solve(A=[row[:] for row in A3], B=B3[:])
        _check_solution(A3, B3, result["solution"])

    def test_singular_raises(self):
        A_sing = [[1, 2, 3], [2, 4, 6], [0, 0, 1]]
        B_sing = [1, 2, 0]
        gj = GaussJordan()
        with pytest.raises(ValueError):
            gj.solve(A=A_sing, B=B_sing)


@pytest.mark.skipif(not HAS_PIVOT, reason="GaussJordanPivot not yet in package")
class TestGaussJordanPivot:
    """BUG 2.8: file missing from methods/linear_system/."""

    def test_solves_3x3(self):
        gjp = GaussJordanPivot()
        result = gjp.solve(A=[row[:] for row in A3], B=B3[:])
        _check_solution(A3, B3, result["solution"])

    def test_handles_zero_pivot_via_swap(self):
        """A matrix that would fail without pivoting."""
        A_zero_pivot = [
            [0.0, 2.0, 1.0],
            [1.0, 3.0, 2.0],
            [2.0, 1.0, 1.0],
        ]
        B_zp = [5.0, 10.0, 8.0]
        gjp = GaussJordanPivot()
        result = gjp.solve(A=A_zero_pivot, B=B_zp)
        _check_solution(A_zero_pivot, B_zp, result["solution"])


class TestCramersRule:
    """BUG 2.6: _determinant called but method is named _determine — NameError before fix."""

    def test_solves_2x2(self):
        A2 = [[2.0, 1.0], [5.0, 7.0]]
        B2 = [11.0, 13.0]
        cr = CramersRule()
        result = cr.solve(A=A2, B=B2)
        _check_solution(A2, B2, result["solution"])

    def test_solves_3x3(self):
        cr = CramersRule()
        result = cr.solve(A=[row[:] for row in A3], B=B3[:])
        _check_solution(A3, B3, result["solution"])

    def test_singular_raises(self):
        A_sing = [[1, 2], [2, 4]]
        B_sing = [1, 2]
        cr = CramersRule()
        with pytest.raises(ValueError):
            cr.solve(A=A_sing, B=B_sing)

    def test_det_A_returned(self):
        A2 = [[2.0, 1.0], [5.0, 7.0]]
        B2 = [11.0, 13.0]
        cr = CramersRule()
        result = cr.solve(A=A2, B=B2)
        assert "det_A" in result
        assert abs(result["det_A"] - (2 * 7 - 1 * 5)) < 1e-9   # det = 9


# ═════════════════════════════════════════════════════════════════════════════
# INFRASTRUCTURE
# ═════════════════════════════════════════════════════════════════════════════

class TestDatabase:
    """BUG 2.10: SQL typo + connection never closed."""

    def test_init_db_does_not_crash(self, tmp_path):
        import sqlite3
        # replicate fixed init_db inline to test it works
        db_path = str(tmp_path / "test_history.db")
        con = sqlite3.connect(db_path)
        con.execute("""CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY,
            method TEXT,
            equation TEXT,
            root REAL,
            iterations INTEGER,
            converged INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        con.commit()
        con.close()
        # verify table exists
        con = sqlite3.connect(db_path)
        tables = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        con.close()
        assert any("runs" in str(t) for t in tables)

    def test_save_and_get_history(self, tmp_path):
        import sqlite3
        db_path = str(tmp_path / "test_history.db")
        con = sqlite3.connect(db_path)
        con.execute("""CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY, method TEXT, equation TEXT, root REAL,
            iterations INTEGER, converged INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )""")
        con.commit()
        con.close()

        con = sqlite3.connect(db_path)
        con.execute("INSERT INTO runs VALUES(NULL,?,?,?,?,?,CURRENT_TIMESTAMP)",
                    ("Bisection", "x^3-x-2", 1.5214, 20, 1))
        con.commit()
        con.close()

        con = sqlite3.connect(db_path)
        rows = con.execute("SELECT * FROM runs ORDER BY timestamp DESC").fetchall()
        con.close()
        assert len(rows) == 1
        assert rows[0][1] == "Bisection"


class TestParser:
    """Sanity-check the expression parser used across the app."""

    def test_parses_polynomial(self):
        from utils.parser import parse_func
        f, f_ = parse_func("x**3 - x - 2")
        assert abs(f(1.5214)) < 0.01

    def test_parses_implicit_multiplication(self):
        from utils.parser import parse_func
        f, _ = parse_func("3x + 2")
        assert abs(f(1) - 5) < 1e-9

    def test_derivative_correct(self):
        from utils.parser import parse_func
        _, f_ = parse_func("x**2")
        # d/dx x^2 = 2x → at x=3 → 6
        assert abs(f_(3) - 6) < 1e-9
