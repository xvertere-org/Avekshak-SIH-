"""
Root conftest.py configuration.

Prevents Windows Application Control policy violation from crashing Python
when an external DLL (pyarrow._compute) is blocked by the host OS policy.
Treats pyarrow as an uninstalled optional dependency via meta_path.
"""

import sys

class _BlockBlockedPyArrow:
    def find_spec(self, fullname, path, target=None):
        if fullname == "pyarrow" or fullname.startswith("pyarrow."):
            raise ModuleNotFoundError(f"No module named {fullname}")
        return None

# Only insert if pyarrow is not already imported
if "pyarrow" not in sys.modules:
    sys.meta_path.insert(0, _BlockBlockedPyArrow())
