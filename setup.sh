#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup.sh — turn-key environment setup for hiddenlayers-predictive-sales
# Usage:  bash setup.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PYTHON=${PYTHON_BIN:-python3}
VENV_DIR=".venv"

echo "═══════════════════════════════════════════════════════"
echo "  hiddenlayers-predictive-sales — environment setup"
echo "═══════════════════════════════════════════════════════"

# 1. Python version check (numeric, not lexicographic)
PY_VER=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "► Python version: $PY_VER"
if ! "$PYTHON" -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)"; then
    echo "ERROR: Python 3.9+ required (found $PY_VER). Set PYTHON_BIN to override."
    exit 1
fi

# 2. Create virtual environment
if [[ ! -d "$VENV_DIR" ]]; then
    echo "► Creating virtual environment at $VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "► Virtual environment already exists at $VENV_DIR — reusing."
fi

# 3. Activate and upgrade pip
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --quiet --upgrade pip wheel

# 4. Install project dependencies
echo "► Installing dependencies from requirements.txt ..."
pip install --quiet -r requirements.txt

# 5. Smoke-test critical imports
echo "► Running import smoke test ..."
python - <<'PYEOF'
from importlib.util import find_spec
import sys
pkgs = ["pandas", "numpy", "sklearn", "torch", "matplotlib",
        "seaborn", "transformers", "sentence_transformers", "tqdm"]
failed = [p for p in pkgs if find_spec(p) is None]
if failed:
    print(f"WARN: missing packages: {failed}", file=sys.stderr)
    sys.exit(1)
print("  All imports OK.")
PYEOF

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Setup complete. Activate with:"
echo "    source $VENV_DIR/bin/activate"
echo ""
echo "  Then run notebooks in order:"
echo "    Phase 1: notebooks/01 → 03"
echo "    Phase 2: notebooks/04 → 06   (GPU recommended for 04)"
echo "    Phase 3: notebooks/07        (GPU recommended)"
echo "═══════════════════════════════════════════════════════"
