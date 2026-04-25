import streamlit as st
import pandas as pd
import numpy as np
from utils import parse_func
from methods.root_finding import (
    Bisection,
    FalsePosition,
    SimpleFixedPoint,
    Newton,
    Secant,
)
from methods.linear_system import (
    GaussElimination,
    LUDecomposition,
    GaussJordan,
    GaussJordanPivot,
    CramersRule,
)

# ==========================================
# Application Configuration
# ==========================================
st.set_page_config(
    page_title="Numerical Analysis Solver",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# Sidebar Navigation & Settings
# ==========================================
with st.sidebar:
    st.title("🧮 Engine Config")
    st.markdown("Select a mathematical domain and algorithm to proceed.")

    category = st.selectbox("Chapter", ["1-Root Finding", "2-Linear Systems"])

    st.divider()

    if category == "1-Root Finding":
        method = st.radio(
            "Algorithm",
            ["Bisection", "False Position", "Simple Fixed Point", "Newton", "Secant"],
        )
    else:
        method = st.radio(
            "Algorithm",
            [
                "Gauss Elimination",
                "LU Decomposition",
                "Gauss Jordan",
                "Gauss Jordan (partial pivot)",
                "Cramer's Rule",
            ],
        )

# ==========================================
# Main UI: Root Finding
# ==========================================
if category == "1-Root Finding":
    st.header(f"Root Finding: {method} Method")

    if method in ["Bisection", "False Position"]:
        st.markdown(
            "Locate the roots of a continuous function $f(x)$ within a bounded interval."
        )
    elif method == "Secant":
        st.markdown("Locate roots using two initial estimates.")
    else:
        st.markdown("Locate roots starting from a single initial guess.")

    with st.container(border=True):
        st.subheader("Algorithm Parameters")

        if method == "Simple Fixed Point":
            func_str = st.text_input(
                "Iteration Function $g(x)$ (where $x = g(x)$)",
                placeholder="sqrt(x + 2)",
            )
        else:
            func_str = st.text_input(
                "Mathematical Function $f(x)$", placeholder="x**3 - x - 2"
            )

        if method in ["Bisection", "False Position", "Secant"]:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if method == "Secant":
                    x_1 = st.number_input("$x_-1$", value=1.0, step=1.0)
                else:
                    xl = st.number_input("$x_l$", value=1.0, step=1.0)
            with col2:
                if method == "Secant":
                    x0 = st.number_input("$x_0$", value=1.0, step=1.0)
                else:
                    xu = st.number_input("$x_u$", value=1.0, step=1.0)
            with col3:
                tol = st.number_input("Tolerance", value=0.0001, format="%.5f")
            with col4:
                max_iter = st.number_input("Max Iterations", value=50, step=1)

        else:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                x0 = st.number_input("Initial Guess ($x_0$)", value=1.5, step=0.1)
            with col2:
                tol = st.number_input("Tolerance", value=0.0001, format="%.5f")
            with col3:
                max_iter = st.number_input("Max Iterations", value=50, step=1)

        if not func_str.strip():
            st.warning("⚠️ Please enter a mathematical function before executing.")
            st.stop()
        f, f_ = parse_func(func_str)

    # 2. Execution and Results
    if st.button("Execute Solver", type="primary", use_container_width=True):
        with st.spinner("Computing iterations..."):
            try:
                if method == "Bisection":
                    solver = Bisection(f, xl, xu, tol=tol, max_iter=max_iter)
                    result = solver.solve()
                elif method == "False Position":
                    solver = FalsePosition(
                        f=f, xl=xl, xu=xu, tol=tol, max_iter=max_iter
                    )
                    result = solver.solve()
                elif method == "Simple Fixed Point":
                    solver = SimpleFixedPoint(g=f, xi=x0, tol=tol, max_iter=max_iter)
                    result = solver.solve()
                elif method == "Newton":
                    solver = Newton(f=f, f_prime=f_, xi=x0, tol=tol, max_iter=max_iter)
                    result = solver.solve()
                elif method == "Secant":
                    solver = Secant(f=f, xi=x0, xi_1=x_1, tol=tol, max_iter=max_iter)
                    result = solver.solve()
            except ValueError as ve:
                st.error(f"Mathematical Error: {ve}")

            # Display results cleanly using Tabs
            tab1, tab2 = st.tabs(["📊 Iteration Table", "📈 Function Plot"])

            with tab1:
                st.metric(
                    label="Converged Root",
                    value=f"{result['root']:.4f}",
                    delta="Success",
                )
                st.dataframe(
                    result["iterations"], use_container_width=True, hide_index=True
                )

            with tab2:
                st.info(
                    "Visualization engine (e.g., Plotly or Matplotlib) will render the function curve here."
                )

# ==========================================
# Main UI: Linear Systems
# ==========================================
elif category == "2-Linear Systems":
    st.header(f"Linear Systems: {method}")
    st.markdown("Solve equations in the form $AX = B$ using matrix operations.")

    # 1. Parameter Configuration
    with st.container(border=True):
        n = st.number_input(
            "System Dimension ($n \\times n$)", min_value=2, max_value=10, value=3
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("**Coefficient Matrix (A)**")
            df_A = pd.DataFrame(
                np.zeros((n, n)), columns=[f"x{i + 1}" for i in range(n)]
            )
            matrix_A = st.data_editor(df_A, use_container_width=True, key="matrix_A")

        with col2:
            st.write("**Constants (B)**")
            df_B = pd.DataFrame(np.zeros((n, 1)), columns=["Value"])
            matrix_B = st.data_editor(df_B, use_container_width=True, key="matrix_B")

    # 2. Execution and Results
    if st.button("Resolve Matrix", type="primary", use_container_width=True):
        with st.spinner("Applying matrix operations..."):
            try:
                # IMPORTANT: Convert Streamlit DataFrames to NumPy float arrays instantly
                A_arr = matrix_A.to_numpy(dtype=float)
                B_arr = matrix_B.to_numpy(dtype=float).flatten()

                # Initialize the correct solver based on the dropdown
                if method == "Gauss Elimination":
                    solver = GaussElimination(A=A_arr, B=B_arr)
                elif method == "LU Decomposition":
                    solver = LUDecomposition(A=A_arr, B=B_arr)
                elif method == "Gauss Jordan":
                    solver = GaussJordan(A=A_arr, B=B_arr)
                elif method == "Gauss Jordan (partial pivot)":
                    solver = GaussJordanPivot(A=A_arr, B=B_arr)
                elif method == "Cramer's Rule":
                    solver = CramersRule(A=A_arr, B=B_arr)

                # Execute the solver
                result = solver.solve()

                st.success("System converged successfully.")

                # ==========================================
                # Display Final Solution Vector (X)
                # ==========================================
                st.subheader("Solution Vector (X)")
                solution_array = result.get("solution", [])

                # Create dynamic columns for x1, x2, x3...
                solution_cols = st.columns(n)
                for i, col in enumerate(solution_cols):
                    val = solution_array[i] if i < len(solution_array) else float("nan")
                    col.metric(label=f"x{i + 1}", value=f"{val:.4f}")

                # ==========================================
                # Display Optional Data (Determinants & Steps)
                # ==========================================

                # If Cramer's Rule was used, show the determinant cleanly
                if "det_A" in result:
                    st.info(f"**Main Determinant:** det(A) = {result['det_A']:.4f}")

                # Display intermediate steps and matrices
                with st.expander("Review Matrix Transformations & History"):
                    # Special rendering for LU Decomposition matrices
                    if method == "LU Decomposition":
                        st.write("**Lower Matrix (L):**")
                        st.dataframe(result.get("L"), use_container_width=True)
                        st.write("**Upper Matrix (U):**")
                        st.dataframe(result.get("U"), use_container_width=True)
                        st.markdown("---")

                    # Standard step-by-step rendering for all other methods
                    steps = result.get("steps", [])
                    if not steps:
                        st.write("No intermediate steps recorded.")
                    else:
                        for step in steps:
                            st.write(f"**{step['description']}**")
                            # Render the matrix state if it exists
                            if step.get("matrix_state") is not None:
                                st.dataframe(
                                    step["matrix_state"], use_container_width=True
                                )

            # ==========================================
            # Professional Error Handling
            # ==========================================
            except ValueError as ve:
                st.error(f"Mathematical Error: {ve}")
            except Exception as e:
                st.error(f"An unexpected error occurred during execution: {e}")
