from __future__ import annotations

import runpy
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
DCF_PAGE = ROOT / "pages" / "02_DCF_Analyzer.py"

if DCF_PAGE.exists():
    runpy.run_path(str(DCF_PAGE), run_name="__main__")
else:
    st.set_page_config(page_title="DCF Analyzer", layout="wide")
    st.error("`pages/02_DCF_Analyzer.py` wurde im Deployment nicht gefunden.")
    st.markdown("Bitte lokal committen/pushen und neu deployen:")
    st.code(
        "git add app.py pages/02_DCF_Analyzer.py\n"
        "git commit -m \"Include DCF page in deployment\"\n"
        "git push",
        language="bash",
    )
