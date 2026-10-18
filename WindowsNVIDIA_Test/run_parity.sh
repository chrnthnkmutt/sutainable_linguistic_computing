#!/bin/bash
# GGUF rows to match the senior's M5 table (central prompt, no thinking, 10 runs per session),
# one hardware per run:
#   unsloth Q4_K_M on GPU: cold start (no warm-up, model unloaded first) + 2 warm sessions
#   qwen3.5:4b on GPU, qwen3.5:4b-cpu on CPU: one repeat session each (first ones: results/2026-10-13_e_central_prompt)
#   unsloth Q4_K_M on CPU (qwen3.5-unsloth-q4km-cpu, num_gpu 0): 2 sessions (extra, not in the senior's table)
# Usage: ./run_parity.sh [results/<dir>]   - never deletes anything; finished sets are skipped.
OUT=${1:-results/$(date +%Y-%m-%d)_f_parity_gguf}
mkdir -p "$OUT"
PY=${PY:-python}   # set PY=/path/to/venv/python to use a venv
export PYTHONIOENCODING=utf-8
UNSLOTH=hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_M
UNSLOTH_CPU=qwen3.5-unsloth-q4km-cpu
run() { name=$1; shift; [ -f "$OUT/$name.xlsx" ] && { echo "skip $name"; return; }
  log="$OUT/log_$name.txt"; [ -f "$log" ] && mv "$log" "$OUT/log_${name}_incomplete_$(date +%H%M%S).txt"
  "$PY" main_ollama.py "$@" --thinking no --runs 10 --output "$OUT/$name.xlsx" > "$log" 2>&1; echo "$(date +%T) $name exit $?"; }
stop_all() { for m in $(ollama ps | awk 'NR>1 {print $1}'); do ollama stop "$m"; done; }

ollama list | grep -q "^$UNSLOTH_CPU" || ollama create "$UNSLOTH_CPU" -f qwen3.5_unsloth_q4km_cpu.modelfile

stop_all
run gpu_unsloth_q4km_cold --model "$UNSLOTH" --hardware nvidia_gpu
run gpu_unsloth_q4km_1    --model "$UNSLOTH" --hardware nvidia_gpu --warmup
run gpu_unsloth_q4km_2    --model "$UNSLOTH" --hardware nvidia_gpu --warmup
stop_all
run gpu_qwen35_4b_repeat  --model qwen3.5:4b --hardware nvidia_gpu --warmup
stop_all
run cpu_qwen35_4b_repeat  --model qwen3.5:4b-cpu --hardware intel_cpu --warmup
stop_all
run cpu_unsloth_q4km_1    --model "$UNSLOTH_CPU" --hardware intel_cpu --warmup
run cpu_unsloth_q4km_2    --model "$UNSLOTH_CPU" --hardware intel_cpu --warmup
stop_all
echo "$(date +%T) ALL DONE"
