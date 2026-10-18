#!/bin/bash
# Same settings as the M5 reports (pboat_sustain_slm_report_*_M5_24G_air.xlsx): temperature 0.0, no thinking,
# central prompt, unlimited max tokens. Our earlier runs used the Modelfile temperature instead
# (qwen3.5:4b = 1.0; unsloth GGUFs have no PARAMETER so Ollama's 0.8 applied).
# 10 runs per session (M5 used 1 run), one hardware per run.
#   Part 1: the 8 M5-table GGUF rows (+ unsloth Q4_K_M CPU)
#   Part 2: every other unsloth quant on GPU and CPU
# Usage: ./run_temp0.sh [results/<dir>]   - never deletes anything; finished sets are skipped.
OUT=${1:-results/$(date +%Y-%m-%d)_h_temp0}
mkdir -p "$OUT"
PY=${PY:-python}   # set PY=/path/to/venv/python to use a venv
export PYTHONIOENCODING=utf-8
run() { name=$1; shift; [ -f "$OUT/$name.xlsx" ] && { echo "skip $name"; return; }
  log="$OUT/log_$name.txt"; [ -f "$log" ] && mv "$log" "$OUT/log_${name}_incomplete_$(date +%H%M%S).txt"
  "$PY" main_ollama.py "$@" --thinking no --temperature 0 --runs 10 --output "$OUT/$name.xlsx" > "$log" 2>&1; echo "$(date +%T) $name exit $?"; }
stop_all() { for m in $(ollama ps | awk 'NR>1 {print $1}'); do ollama stop "$m"; done; }
U=hf.co/unsloth/Qwen3.5-4B-GGUF

# Part 1 - M5 table rows
stop_all
run cpu_qwen35_4b         --model qwen3.5:4b-cpu --hardware intel_cpu --warmup
stop_all
run gpu_qwen35_4b         --model qwen3.5:4b --hardware nvidia_gpu --warmup
run gpu_qwen35_4b_repeat  --model qwen3.5:4b --hardware nvidia_gpu --warmup
stop_all
run gpu_unsloth_q4-k-m_cold   --model $U:Q4_K_M --hardware nvidia_gpu
run gpu_unsloth_q4-k-m        --model $U:Q4_K_M --hardware nvidia_gpu --warmup
run gpu_unsloth_q4-k-m_repeat --model $U:Q4_K_M --hardware nvidia_gpu --warmup
stop_all
run cpu_unsloth_q4-k-m    --model qwen3.5-unsloth-q4km-cpu --hardware intel_cpu --warmup

# Part 2 - other quants
for q in UD-Q2_K_XL Q3_K_M Q5_K_M Q6_K Q8_0 BF16; do
  l=$(echo $q | tr 'A-Z_' 'a-z-')
  stop_all
  run gpu_unsloth_$l --model "$U:$q" --hardware nvidia_gpu --warmup
  stop_all
  run cpu_unsloth_$l --model "qwen3.5-unsloth-$l-cpu" --hardware intel_cpu --warmup
done
stop_all
echo "$(date +%T) TEMP0 DONE"
