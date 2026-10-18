# bench: Thai SLM inference energy (Ollama GGUF / MLX)

Measures energy, speed and quality of Qwen3.5-4B across quantization levels, thinking on/off,
thinking budgets and Thai vs. English prompts.

Energy is read with **Zeus** (`zeus_meter.py`): NVIDIA/AMD GPU (NVML energy counter read at request
boundaries, plus an instantaneous-power trace), CPU RAPL on Linux, Apple Silicon SoC
(zeus-apple-silicon, no sudo) and, on Windows, CPU package energy from the Windows Energy Meter
(`win_emi.py`: RAPL via EMI, picowatt-hours, no admin rights).

```
configs/        prompts.yaml (Thai/English workload) and run configs per backend
bench.py        runs a config -> results/<run>/results.jsonl, blocks.jsonl, traces/, system.json
analyze.py      results -> analysis/ (summary CSVs, report.md, figures)
zeus_meter.py   Zeus energy timeline (counters read only at request boundaries)
win_emi.py      Windows CPU package energy (Energy Meter Interface)
```

The NVML energy counter is read only at request boundaries: polling it continuously raised idle GPU
power on the RTX 5070 Ti by +3.7 W at 1 Hz and +50.8 W at 10 Hz. The counter updates every ~100 ms, so
a 0.25 s tail is waited after each request and short requests are aggregated (`sum J / sum tokens`).

## Protocol (per quant)

1. Download outside measurement; unload other models; load + warm up each mode (not measured).
2. Per mode: wait until GPU power is stable, measure idle power for 30 s.
3. Run every prompt × language × repeat (5 repeats, seed = 1000 + repeat, order shuffled
   per repeat). Each request: counter mark → streamed generation → 0.25 s tail → counter mark.
4. Record total and net (minus idle × duration) energy, phase split, tokens, TTFT, tok/s,
   peak power, GPU temperature/clock, full answer and thinking text, `done_reason`.

Sampling follows Qwen's recommendations (thinking: T=0.6, top-p 0.95; non-thinking: T=0.7,
top-p 0.8; top-k 20, presence penalty 1.5). Greedy decoding (T=0) made thinking loop to the
token cap on 2 of 3 test prompts. Thinking is capped at 8,192 tokens; hitting the cap is
recorded (`done_reason = length`) and reported as the finish rate.

## Run on Windows + NVIDIA (Ollama)

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
ollama serve   # already running as a service on Windows
.venv/Scripts/python bench.py --config configs/ollama_qwen35_4b.yaml --out results/ollama_qwen35_4b
.venv/Scripts/python analyze.py results/ollama_qwen35_4b
```

Do not use the PC while it runs (background load contaminates the measurement). Interrupt any
time; rerunning the same command resumes. `--quants Q4_K_M --modes no_think --repeats 1` for a
quick check, `--dry-run` to see the plan size.

## Thinking effort (budget forcing)

`configs/ollama_qwen35_4b_budget.yaml` expands `think_budgets: [0, 256, …, 8192]` into one mode
per budget. Qwen3.5 has no built-in effort levels, so `OllamaBackend.chat_budget` uses
budget forcing through Ollama's raw `/api/generate`:

1. render the ChatML prompt and open `<think>`; generate at most N tokens with stop `</think>`;
2. append the (possibly cut) thinking plus `\n</think>\n\n` and generate the answer
   (≤ `num_predict` tokens). Phase 2 shares phase 1's prefix, so Ollama reuses the KV cache
   (≈20 ms prefill) and the forced continuation adds no prompt-processing energy.

Budget 0 uses the template's empty thinking block. Every request therefore produces an answer;
rows record `think_tokens`, `answer_tokens` and `think_done_reason` (`length` = cut by the
budget). The same sampling settings are used for every budget so only the budget varies.

```bash
.venv/Scripts/python bench.py --config configs/ollama_qwen35_4b_budget.yaml --out results/ollama_qwen35_4b_budget
.venv/Scripts/python analyze.py results/ollama_qwen35_4b_budget   # budget_summary.csv, fig8-10
```

## Run on Apple Silicon (MLX)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt mlx-lm
for q in bf16 8bit 6bit 4bit 3bit mixed_2_6; do huggingface-cli download mlx-community/Qwen3.5-4B-$q; done
mlx_lm.server --host 127.0.0.1 --port 8080 &
.venv/bin/python bench.py --config configs/mlx_qwen35_4b.yaml --out results/mlx_qwen35_4b
.venv/bin/python analyze.py results/mlx_qwen35_4b
```

Check first with `--quants 4bit --repeats 1` that (a) the Apple Silicon energy domains are discovered and (b) thinking toggles: rows with `think=false` should have
`think_chars = 0`. If your mlx-lm version ignores `chat_template_kwargs`, set
`thinking_kwarg` accordingly or upgrade mlx-lm. Qwen3.5 checkpoints are multimodal; if
`mlx_lm` cannot load one, serve it with `mlx_vlm.server` instead (same OpenAI API).

## Outputs

`results/<run>/results.jsonl` - one row per request (energy per domain: total / net / per phase /
trace, tokens, timings, GPU state, answer and thinking text). `blocks.jsonl` - per quant × mode
block totals for cross-checking. `traces/*.csv.gz` - power timelines.
`analysis/` - `summary_by_condition.csv`, `summary_by_task.csv`, `th_vs_en.csv`,
`phase_split.csv`, `block_crosscheck.csv`, `report.md`, `fig*.png`.

Automatic quality scoring covers QA (gold answer present) and sentiment (exact label);
summaries and jokes are kept in full for human / LLM-judge evaluation.
