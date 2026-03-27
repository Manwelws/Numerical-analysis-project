import streamlit as st

st.set_page_config(
    page_title="Numerical Solver",
    page_icon="",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={"About": "https://www.linkedin.com/in/manwel-wasfy-537546323/"},
)

if "calculated" not in st.session_state:
    st.session_state.calculated = False
