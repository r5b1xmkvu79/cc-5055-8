#!/usr/bin/env bash
# Graphical entry point for the dynamic-scene shot solver.
# The GUI is the documented primary interface; the CLI is a headless helper.
set -euo pipefail
cd "$(dirname "$0")"
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" exec ./.venv/bin/python -I -m dynamicsolver.gui.app "$@"
