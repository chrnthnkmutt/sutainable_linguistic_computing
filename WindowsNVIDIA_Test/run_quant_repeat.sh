#!/bin/bash
# Second pass of the quantization sweep (run after run_quant.sh):
#   GPU: one repeat session per quant (10 runs), so every quant has 2 GPU sessions like Q4_K_M
#   CPU: repeat UD-Q2_K_XL only (first session had a 67 W idle baseline and one run looping to the 4,096 context)
# The first GPU UD-Q2_K_XL session is kept but not used: its idle baseline was 166 W (another app was using the GPU).
# Usage: ./run_quant_repeat.sh [results/<dir>]   - never deletes anything; finished sets are skipped.
OUT=${1:-results/2026-10-14_g_quant_sweep}
mkdir -p "$OUT"
PY=${PY:-python}   # set PY=/path/to/venv/python to use a venv
export PYTHONIOENCODING=utf-8
run() { name=$1; shift; [ -f "$OUT/$name.xlsx" ] && { echo "skip $name"; return; }
  log="$OUT/log_$name.txt"; [ -f "$log" ] && mv "$log" "$OUT/log_${name}_incomplete_$(date +%H%M%S).txt"
  "$PY" main_ollama.py "$@" --thinking no --runs 10 --output "$OUT/$name.xlsx" > "$log" 2>&1; echo "$(date +%T) $name exit $?"; }
stop_all() { for m in $(ollama ps | awk 'NR>1 {print $1}'); do ollama stop "$m"; done; }

for q in UD-Q2_K_XL Q3_K_M Q5_K_M Q6_K Q8_0 BF16; do
  l=$(echo $q | tr 'A-Z_' 'a-z-')
  stop_all
  run gpu_unsloth_${l}_repeat --model "hf.co/unsloth/Qwen3.5-4B-GGUF:$q" --hardware nvidia_gpu --warmup
done
stop_all
run cpu_unsloth_ud-q2-k-xl_repeat --model qwen3.5-unsloth-ud-q2-k-xl-cpu --hardware intel_cpu --warmup
stop_all
echo "$(date +%T) REPEAT DONE"
