import streamlit as st


def gets_input_for(method_name: str, f, f_):
    if method_name in ("Bisection", "False Position"):
        col1, col2 = st.columns(2)
        xl = col1.number_input("xl", value=0.0)
        xu = col2.number_input("xu", value=1.0)
        return {"f": f, "xl": xl, "xu": xu}

    elif method_name == "Simple Fixed Point":
        x0 = st.number_input("x0", value=0.0)
        return {"f": f, "xi": x0}

    elif method_name == "Newton":
        x0 = st.number_input("x0", value=0.0)
        return {"f": f, "f_prime": f_, "xi": x0}
