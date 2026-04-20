from methods.root_finding import Secant
from methods.linear_system import (
    GaussElimination,
    GaussJordan,
    LUDecomposition,
    CramersRule,
)

"""
Quick smoke-test for all new implementations.
Run from the project root:  python test_new_methods.py
"""
import math, sys


# ── Helpers ──────────────────────────────────────────────────────────────────
def ok(name):
    print(f"  ✓  {name}")


def fail(name, e):
    print(f"  ✗  {name}: {e}")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# Ch1 — Secant
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Root Finding ─────────────────────────────────────────────────────")


f = lambda x: x**3 - x - 2  # root ≈ 1.5214

try:
    res = Secant(f=f, xi_1=1.0, xi=2.0, tol=1e-6).solve()
    assert res["converged"], "Did not converge"
    assert abs(res["root"] - 1.5213797) < 1e-4
    ok(f"Secant  root={res['root']:.7f}  iters={len(res['iterations'])}")
except Exception as e:
    fail("Secant", e)


# ═══════════════════════════════════════════════════════════════════════════
# Ch2 — Linear Systems
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Linear Systems ───────────────────────────────────────────────────")


# System:  2x + y - z = 8
#         -3x - y + 2z = -11
#         -2x + y + 2z = -3
# Expected: x=2, y=3, z=-1

A = [
    [2, 1, -1],
    [-3, -1, 2],
    [-2, 1, 2],
]
B = [8, -11, -3]
EXPECTED = [2.0, 3.0, -1.0]


def check(name, method_cls, **kwargs):
    try:
        res = method_cls(**kwargs).solve(A=[row[:] for row in A], B=B[:])
        x = res["solution"]
        for i, (got, exp) in enumerate(zip(x, EXPECTED)):
            assert abs(got - exp) < 1e-6, f"x[{i}]={got:.6f} expected {exp}"
        ok(f"{name:30s}  x={[round(v, 4) for v in x]}")
    except Exception as e:
        fail(name, e)


check("GaussElimination", GaussElimination)
check("GaussJordan", GaussJordan)
check("LUDecomposition", LUDecomposition)
check("CramersRule", CramersRule)

print("\nAll tests passed ✓\n")
