"""
═══════════════════════════════════════════════════════════════════════════════
Utilities
═══════════════════════════════════════════════════════════════════════════════
Shared helper functions and config used by all other scripts.
═══════════════════════════════════════════════════════════════════════════════
"""

from pathlib import Path
from collections import defaultdict

""""
═══════════════════════════════════════════════════════════════════════════════
Configuration 
═══════════════════════════════════════════════════════════════════════════════
"""

CLASS_NAMES = [
    "background",
    "normal",
    "diabetic",
    "pressure",
    "surgical",
    "venous",
]

CLASS_FOLDER_HINTS = {
    "background": ["BG", "bg", "Bg", "background"],
    "normal":     ["N",  "normal",   "Normal"],
    "diabetic":   ["D",  "diabetic", "Diabetic"],
    "pressure":   ["P",  "pressure", "Pressure"],
    "surgical":   ["S",  "surgical", "Surgical"],
    "venous":     ["V",  "venous",   "Venous"],
}

BOX_COVERAGE = 0.90
TRAIN_RATIO  = 0.70
VALIDATE_RATIO    = 0.20
TEST_RATIO   = 0.10

# ── Print helpers ────────────────────────────────
def print_header(t): print(f"\n{'='*55}\n   {t}\n{'='*55}")
def print_success(m): print(f"  ✅ {m}")
def print_warn(m):    print(f"  ⚠️  {m}")
def print_error(m):   print(f"  ❌ {m}")
def print_info(m):    print(f"     {m}")

"""
═══════════════════════════════════════════════════════════════════════════════
Weights finder 
═══════════════════════════════════════════════════════════════════════════════
"""

def find_best_weights(backup="best_model.pt"):
    if Path(backup).exists():
        return backup
    if Path("runs").exists():
        matches = sorted(Path("runs").rglob("best.pt"))
        if matches:
            return str(matches[-1])
    return None
