#!/bin/bash
# Team central prompt (main.py DEFAULT_MESSAGES: "You are a helpful assistant." +
# "กระบวนการทางวิทยาศาสตร์มีกี่ขั้นตอนอะไรบ้าง?"), one hardware per run:
#   GPU model (qwen3.5:4b)      -> measure NVIDIA GPU only
#   CPU model (qwen3.5:4b-cpu)  -> measure CPU only
# Usage: ./run_central.sh [results/<dir>]   - never deletes anything; finished sets are skipped.
OUT=${1:-results/$(date +%Y-%m-%d)_e_central_prompt}
mkdir -p "$OUT"
PY=${PY:-python}   # set PY=/path/to/venv/python to use a venv
export PYTHONIOENCODING=utf-8
run() { name=$1; shift; [ -f "$OUT/$name.xlsx" ] && { echo "skip $name"; return; }
  log="$OUT/log_$name.txt"; [ -f "$log" ] && mv "$log" "$OUT/log_${name}_incomplete_$(date +%H%M%S).txt"
  "$PY" main_ollama.py "$@" --output "$OUT/$name.xlsx" > "$log" 2>&1; echo "$(date +%T) $name exit $?"; }
run gpu_nothink --model qwen3.5:4b --hardware nvidia_gpu --thinking no  --runs 10 --warmup
run gpu_think   --model qwen3.5:4b --hardware nvidia_gpu --thinking low --runs 5  --warmup --max-tokens 8192
ollama stop qwen3.5:4b
run cpu_nothink --model qwen3.5:4b-cpu --hardware intel_cpu --thinking no  --runs 10 --warmup
run cpu_think   --model qwen3.5:4b-cpu --hardware intel_cpu --thinking low --runs 3  --warmup --max-tokens 8192
ollama stop qwen3.5:4b-cpu
echo "$(date +%T) ALL DONE"
