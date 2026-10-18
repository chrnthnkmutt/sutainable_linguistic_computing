#!/usr/bin/env bash
# Full Windows/NVIDIA run: Ollama quant ladder, then analysis.
# Resumable: rerun the same script after an interruption.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/Scripts/python
[ -x "$PY" ] || PY=.venv/bin/python
export PYTHONIOENCODING=utf-8

# 1) benchmark
"$PY" -W ignore bench.py --config configs/ollama_qwen35_4b.yaml --out results/ollama_qwen35_4b

# 2) analysis
"$PY" -W ignore analyze.py results/ollama_qwen35_4b
echo "RUN_ALL DONE"
