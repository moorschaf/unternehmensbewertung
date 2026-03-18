import streamlit as st

st.set_page_config(page_title="Financial Models", layout="wide")

st.title("Financial Models Workspace")
st.caption("Zwei getrennte Apps in einer Streamlit-Multipage-Struktur")

left, right = st.columns(2)

with left:
    st.subheader("Option Pricer")
    st.write("Black-Scholes/Binomial Option Pricing mit Greeks, IV, Kurven und Heatmaps.")
    st.page_link("pages/01_Option_Pricer.py", label="Option Pricer oeffnen")

with right:
    st.subheader("DCF Analyzer")
    st.write("DCF-Analyse fuer US-Unternehmen mit SEC-Historie, CAPM-WACC und Terminal Value.")
    st.page_link("pages/02_DCF_Analyzer.py", label="DCF Analyzer oeffnen")
