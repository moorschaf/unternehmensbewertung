import streamlit as st


st.set_page_config(page_title="DCF Analyzer", layout="wide")

try:
    st.switch_page("pages/02_DCF_Analyzer.py")
except Exception:
    st.title("DCF Analyzer")
    st.error("Die DCF-Seite konnte nicht geoeffnet werden.")
    st.info("Setze in Streamlit Cloud alternativ Main file path direkt auf `pages/02_DCF_Analyzer.py`.")
