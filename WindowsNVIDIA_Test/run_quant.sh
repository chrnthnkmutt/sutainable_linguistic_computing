#!/bin/bash
# Quantization sweep with the central prompt (no thinking, 10 runs per session), one hardware per run.
# Unsloth Qwen3.5-4B GGUF: UD-Q2_K_XL, Q3_K_M, Q5_K_M, Q6_K, Q8_0, BF16
# (Q4_K_M is already in results/2026-10-14_f_parity_gguf).
#   GPU: hf.co/unsloth/Qwen3.5-4B-GGUF:<quant>          --hardware nvidia_gpu
#   CPU: qwen3.5-unsloth-<quant>-cpu (num_gpu 0)         --hardware intel_cpu
# Usage: ./run_quant.sh [results/<dir>]   - never deletes anything; finished sets are skipped.
OUT=${1:-results/$(date +%Y-%m-%d)_g_quant_sweep}
mkdir -p "$OUT"
PY=${PY:-python}   # set PY=/path/to/venv/python to use a venv
export PYTHONIOENCODING=utf-8
run() { name=$1; shift; [ -f "$OUT/$name.xlsx" ] && { echo "skip $name"; return; }
  log="$OUT/log_$name.txt"; [ -f "$log" ] && mv "$log" "$OUT/log_${name}_incomplete_$(date +%H%M%S).txt"
  "$PY" main_ollama.py "$@" --thinking no --runs 10 --output "$OUT/$name.xlsx" > "$log" 2>&1; echo "$(date +%T) $name exit $?"; }
stop_all() { for m in $(ollama ps | awk 'NR>1 {print $1}'); do ollama stop "$m"; done; }

for q in UD-Q2_K_XL Q3_K_M Q5_K_M Q6_K Q8_0 BF16; do
  l=$(echo $q | tr 'A-Z_' 'a-z-')
  cpu=qwen3.5-unsloth-$l-cpu
  ollama list | grep -q "^$cpu" || ollama create "$cpu" -f "modelfiles_quant/qwen3.5_unsloth_${l}_cpu.modelfile"
  stop_all
  run gpu_unsloth_$l --model "hf.co/unsloth/Qwen3.5-4B-GGUF:$q" --hardware nvidia_gpu --warmup
  stop_all
  run cpu_unsloth_$l --model "$cpu" --hardware intel_cpu --warmup
done
stop_all
echo "$(date +%T) ALL DONE"
