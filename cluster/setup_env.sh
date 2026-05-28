#!/bin/bash
# One-time setup: create venv and install dependencies on Hábrók.
#
# Usage:
#   ssh habrok
#   cd <project-dir>
#   bash cluster/setup_env.sh
#
# Before running, find available modules:
#   module spider Python
#   module spider Graphviz
#   module spider Yosys
#
# Then set the variables below to match what's available.

set -euo pipefail

# ── Adjust these to match available modules on Hábrók ──────────────────
PYTHON_MODULE="Python/3.11.3-GCCcore-12.3.0"   # run: module spider Python
GRAPHVIZ_MODULE="Graphviz/8.1.0-GCCcore-12.3.0" # run: module spider Graphviz
# YOSYS_MODULE=""  # uncomment if yosys is available as module
# ───────────────────────────────────────────────────────────────────────

VENV_DIR=".venv_habrok"

echo "=== Loading modules ==="
module purge
module load "${PYTHON_MODULE}"
module load "${GRAPHVIZ_MODULE}"
module list

echo ""
echo "=== Creating virtual environment: ${VENV_DIR} ==="
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo ""
echo "=== Installing pip dependencies ==="
pip install --upgrade pip wheel

# Install everything except z3log (needs --no-deps)
grep -v '^z3log\|^#\|^$' requirements.txt | pip install -r /dev/stdin

# z3log without its broken pygraphviz dependency declaration
pip install --no-deps z3log==2.2.13

# pygraphviz (needs graphviz C headers from the loaded module)
pip install pygraphviz

echo ""
echo "=== Verifying installation ==="
python3 -c "import z3; print(f'z3 version: {z3.get_version_string()}')"
python3 -c "import pygraphviz; print('pygraphviz OK')"
python3 -c "import networkx; print('networkx OK')"
python3 -c "import pandas; print('pandas OK')"

echo ""
echo "=== Setup complete ==="
echo "Venv at: $(pwd)/${VENV_DIR}"
echo ""
echo "If Yosys is not available as a module, you may need to:"
echo "  1. Install from source: https://github.com/YosysHQ/yosys"
echo "  2. Or use: module spider Yosys  (to check)"
