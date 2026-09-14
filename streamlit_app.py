"""
Avekshak — SIH26054 Aero Engine Digital Twin & PHM System
Streamlit Cloud Entrypoint.
"""

import sys
from pathlib import Path

# Ensure repository root is in sys.path
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dashboard.app import main

if __name__ == "__main__":
    main()
