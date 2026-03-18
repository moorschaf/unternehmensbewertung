from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DCF_PAGE = ROOT / "pages" / "02_DCF_Analyzer.py"

runpy.run_path(str(DCF_PAGE), run_name="__main__")
