import streamlit as st

st.set_page_config(page_title="DCF Analyzer", layout="wide")

st.title("DCF Analyzer")
st.caption("US-Unternehmensbewertung mit SEC-Historie, CAPM-WACC und 5Y+Terminal-Value")

st.write("Diese Deployment-Variante enthaelt nur den DCF-Bereich.")
st.page_link("pages/02_DCF_Analyzer.py", label="DCF Analyzer oeffnen")
