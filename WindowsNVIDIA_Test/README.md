# WindowsNVIDIA_Test

Sustain_SLM on a Windows PC with an NVIDIA GPU: Windows 11, AMD Ryzen 5 7500F, RTX 5070 Ti 16 GB, 32 GB RAM,
Ollama 0.34.4, Qwen3.5-4B (GGUF). Same tool as `AppleSilicon_Test`, plus the Windows fixes listed below.
`main.py` is the team version merged with these fixes; only `main_ollama.py` (Ollama) was run on this PC.
`energy_bench/` holds the quantization / thinking-budget / Thai-vs-English benchmark (see its README).
Rules: one device per session (GPU model on the GPU, CPU-only model on the CPU), PC idle during runs.
Results are not committed (`results/` is ignored).

## Sustain_SLM

Measures the resource and energy cost of running a small language model and writes the results to an Excel report.

| Entry point | Model served by | Calls it via |
|---|---|---|
| `main.py` | Foundry Local | `foundry_local_sdk` |
| `main_ollama.py` | Ollama | OpenAI-compatible API (`http://localhost:11434/v1`) |

Both use the same hardware monitors (`monitors/`) and the same report writer (`report.py`).

## What changed in this version (Watin, 12 Oct 2026)

Tested on Windows 11 / AMD Ryzen 5 7500F / RTX 5070 Ti with Ollama 0.34.4 and `qwen3.5:4b`.

| # | Problem in the first version | Effect measured | Fix |
|---|---|---|---|
| 1 | Default endpoint `localhost` | Windows tries IPv6 first: TTFT 2.3 s instead of 0.1 s, +50% GPU energy per request, 30 instead of 88 tok/s | Default `http://127.0.0.1:11434/v1` |
| 2 | No CPU energy on Windows, AMD CPUs rejected | CPU-model runs reported only the idle GPU as "Total energy" | CPU energy on **Windows (Intel and AMD)** through the Windows Energy Meter Interface (RAPL, no admin, no driver; `monitors/windows_emi.py`); AMD accepted on Linux RAPL too |
| 3 | Idle baseline taken right after the warm-up | GPU still in a high power state: 47 W instead of ~29 W, so net energy came out too low | Wait until power settles (`--settle-max`, default 90 s), then 15 s baseline |
| 4 | `temperature 0` sent by default | Qwen3.5 thinking loops until the token cap (1,024 of 1,024 tokens were reasoning) | Temperature not sent unless `--temperature` is given, so the Modelfile values apply (qwen3.5: T=1, top_p 0.95, top_k 20; unsloth GGUFs have none, so Ollama's 0.8). For no-thinking runs comparable with the M5 reports pass `--temperature 0` (`run_temp0.sh`) |
| 5 | Tokens/sec included the time to first token | 88 tok/s reported vs 172 tok/s actual decode | New column **Generation tokens/sec**; Tokens/sec documented as end-to-end |
| 6 | Net energy floored at 0 | Biases averages upward | Not floored |
| 7 | Memory monitor scanned every process every 0.25 s | ~1.4 s per scan on Windows: keeps a CPU core busy (inflates CPU energy) and adds ~3 s per run | Watched processes resolved once per run |
| 8 | Ollama's model process `llama-server` not counted | "Process RAM" 253 MB while the model used 5.6 GB | `llama-server` added to the watched processes |
| 9 | Energy window ended at the last token | NVML counter updates every ~100 ms, the end of short runs was missed | 0.3 s tail (`--tail-seconds`) |
| 10 | One prompt per session | | `--prompt-file` (JSON `{id, system, user}` or one prompt per line), new columns **Prompt ID**, **Finish reason**, **Response** |
| 11 | Silent when the measured hardware is not where the model runs | | Note with the model placement from Ollama `/api/ps` (e.g. "100% CPU") and a warning if that hardware has no energy reading |
| 12 | Default conversation was the World Series example | Not the team's central prompt | Default is now the central prompt: system "You are a helpful assistant." + user "กระบวนการทางวิทยาศาสตร์มีกี่ขั้นตอนอะไรบ้าง?" (13 Oct) |
| 13 | Model placement note missing for local tags | Ollama lists them as `name:latest` | Lookup also tries `:latest` |

`summarize.py` turns the reports into one table (`results/summary.md`), with accuracy for the QA and
sentiment prompts from `prompts/gold.json`. `prompts/prompts_th.json` / `prompts_en.json` are the
team's Thai/English workload (20 prompts: QA, summarization, sentiment, humor).

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs the right packages per platform:
`psutil`, `openpyxl`, `py-cpuinfo`, `openai` (Ollama only), `nvidia-ml-py` (not on macOS),
`zeus-apple-silicon` (Apple Silicon Macs only).

Also needed:
- **main.py:** `foundry_local_sdk` (keep your existing install).
- **main_ollama.py:** Ollama installed and running, plus a pulled model: `ollama pull <model>`.

## 2. Hardware support

| Option | Utilisation (%) | Power / energy | Needs |
|---|---|---|---|
| Intel/AMD CPU | psutil | RAPL: Linux powercap (needs `sudo`) or **Windows Energy Meter** (no admin) | Intel or AMD CPU |
| NVIDIA GPU | NVML | NVML energy counter | NVIDIA driver, `nvidia-ml-py` |
| Apple Silicon CPU | psutil | IOReport via `zeus-apple-silicon` (no sudo) | Apple Silicon Mac |
| Apple Silicon GPU | `ioreg` | IOReport via `zeus-apple-silicon` (no sudo) | Apple Silicon Mac |

- Hardware that is not present is shown as unavailable, with the reason, and cannot be selected.
- You can select one CPU and one GPU per session (the report has one CPU block and one GPU block).
- On Windows the CPU package energy comes from the Energy Meter performance counters
  (`Get-Counter -ListSet "Energy Meter"` lists them). Package only: no separate DRAM value.
- Selecting hardware only **measures** it. The model must actually run there (choose a GPU variant/model for GPU runs).

## 3. Run: Foundry Local (`main.py`)

Interactive:

```bash
python main.py
```

You are asked, in order: hardware, thinking level, model alias, number of runs, warm-up run.

Examples:

```bash
# fully scripted, 3 measured runs after a warm-up, no thinking
python main.py --hardware intel_cpu,nvidia_gpu --alias "qwen3.5-2b-generic-cpu:3" --runs 3 --warmup --thinking no

# Apple Silicon
python main.py --hardware apple_cpu,apple_gpu --runs 3 --warmup

# Intel RAPL on Linux needs root
sudo -E env "PATH=$PATH" python main.py --hardware intel_cpu

# test the whole pipeline without loading any model
python main.py --dry-run
```

Change the default model aliases in the `MODEL_ALIASES` dict at the top of `main.py`
(`"cpu"` is used for CPU-only selections, `"gpu"` when a GPU is selected).

## 4. Run: Ollama (`main_ollama.py`)

```bash
ollama serve                       # if it is not already running
ollama pull qwen3:1.7b
python main_ollama.py
```

CPU-only copy of a model (team Modelfile, see `qwen3.5_cpu.modelfile`):

```bash
ollama create qwen3.5:4b-cpu -f qwen3.5_cpu.modelfile
python main_ollama.py --model qwen3.5:4b-cpu --hardware all --prompt-file prompts/prompts_th.json --runs 3 --warmup
```

Examples:

```bash
python main_ollama.py --model qwen3:1.7b --hardware nvidia_gpu,intel_cpu --runs 3 --warmup --thinking low
python main_ollama.py --hardware apple_cpu,apple_gpu --model gpt-oss:20b --thinking high --max-tokens 512
python main_ollama.py --base-url http://192.168.1.20:11434/v1 --model qwen3:1.7b   # remote server
python main_ollama.py --dry-run                                                      # no Ollama needed
```

Notes:
- If you omit `--model`, the script lists the models Ollama has and lets you pick.
- The first request makes Ollama load the model, which inflates time and energy. The warm-up run (default "yes" in the prompt) keeps that out of the results.
- For a remote Ollama server, hardware metrics come from the machine **running this script**, not the server.
- Ollama runs as its own process, so its memory is added to "Process RAM".
- The endpoint can also be set with the `OLLAMA_BASE_URL` environment variable.

## 5. All options

### Both scripts

| Option | Default | Meaning |
|---|---|---|
| `--hardware LIST` | menu | Comma list of `intel_cpu`, `nvidia_gpu`, `apple_cpu`, `apple_gpu`, or `all`. Skips the menu. |
| `--thinking LEVEL` | menu (default **No**) | `no`, `low`, `medium`, `high`. |
| `--runs N` | prompt (1) | Number of measured runs. |
| `--warmup` | prompt | Do one un-measured run first. |
| `--prompt TEXT` | built-in conversation | Single user prompt replacing the default conversation. |
| `--idle-seconds S` | 15 | Idle baseline length before the runs. `0` skips it (net-energy columns become `N/A`). |
| `--settle-max S` | 90 | Wait up to S seconds for power to stop falling before the idle baseline. |
| `--tail-seconds S` | 0.3 | Keep measuring S seconds after the last token. |
| `--prompt-file FILE` | none | Prompt set; every prompt is run `--runs` times. |
| `--interval S` | 0.5 | Sampling interval for usage and power. |
| `--output FILE` | see below | Excel report path. |
| `--dry-run` | off | Simulated inference (CPU burn and memory ballast) to test the pipeline. |

### `main.py` only

| Option | Default | Meaning |
|---|---|---|
| `--alias NAME` | prompt (from `MODEL_ALIASES`) | Foundry Local model variant. |

Default output: `sustain_slm_report.xlsx` next to the script.

### `main_ollama.py` only

| Option | Default | Meaning |
|---|---|---|
| `--model TAG` | menu | Ollama model tag, e.g. `qwen3:1.7b`. |
| `--base-url URL` | `http://127.0.0.1:11434/v1` | OpenAI-compatible endpoint (avoid `localhost` on Windows). |
| `--api-key KEY` | `ollama` | Ignored by Ollama, required by the client. |
| `--max-tokens N` | unlimited | Cap generated tokens, useful for comparing runs fairly. |
| `--temperature T` | not sent | Sampling temperature; by default the model's Modelfile value applies. |

Default output: `sustain_slm_report_ollama.xlsx` next to the script.

## 6. Thinking levels

| Level | Ollama (`main_ollama.py`) | Foundry Local (`main.py`) |
|---|---|---|
| No | `reasoning_effort=none` | Prompt hint `Reasoning: none` + `/no_think` |
| Low / Medium / High | `reasoning_effort=low/medium/high` | Prompt hint `Reasoning: low/medium/high` |

- **Ollama:** the script checks the model's thinking support and warns if the level won't do what you asked
  (on/off-only models treat Low, Medium and High as "on"; models without thinking ignore the level).
- **Foundry Local:** the SDK has no thinking setting, so this is only a request in the prompt. Models such as gpt-oss follow it;
  others may ignore it. The "Thinking method" column states this.
- Check the run notes: **Reasoning chunks streamed** (Ollama) tells you whether the model actually thought.

## 7. The report (`.xlsx`)

A new row is appended on every run, so you can compare machines, models and thinking levels in one file.

| Sheet | Content |
|---|---|
| **Summary** | One row per run: hardware, thinking level, duration, time to first token, tokens, tokens/sec, memory, CPU and GPU (name, usage change, average and peak usage, power, energy, net energy vs idle), DRAM / GPU SRAM / Neural Engine energy, total energy (J and Wh), average total power, **energy per token**, notes. |
| **Samples** | Time series of usage, power, RAM and VRAM for charting. |
| **Statistics** | Mean and standard deviation across the runs of the latest session. |
| **Environment** | OS, library versions, detected hardware, measurement backends, idle power, warnings, metric definitions. |

How the numbers are defined:
- **Usage change (% -> %):** first sample -> last sample during inference. Average and peak come from the samples.
- **Total energy:** CPU + GPU energy of the selected hardware. DRAM, GPU SRAM and Neural Engine energy are shown separately and are not included.
- **Net energy vs idle:** energy minus (idle average power x measured window), not floored.
- **Tokens:** `main.py` counts streamed chunks (approximate). `main_ollama.py` uses Ollama's `completion_tokens`.
- **Time to first token:** includes reasoning tokens if the model emits them.
- **N/A:** not measurable on this machine; the reason is in the Notes column. Values are never estimated.
- If the xlsx is open in Excel and cannot be saved, the report is written to a timestamped copy instead.
- If a report from an older version has different columns, new rows go to a `Summary (v2)` sheet.

## 8. Tips for fair comparisons

- Close other heavy apps; leave the machine idle during the baseline.
- Use the same prompt, `--max-tokens` and `--thinking` level when comparing models or hardware.
- Use `--warmup` and several `--runs`; check the Statistics sheet for the spread.
- On laptops, plug in power and note the power mode in your results.

## 9. Files

```
main.py                 Foundry Local entry point
main_ollama.py          Ollama entry point
report.py               Excel writer
monitors/               one monitor per hardware (+ memory)
requirements.txt
monitors/windows_emi.py CPU energy on Windows (Energy Meter / RAPL)
summarize.py            reports -> results/summary.md (+ csv)
prompts/                Thai/English prompt sets and gold answers
run_central.sh          the Qwen3.5-4B runs with the central prompt (results/2026-10-13_e_central_prompt)
run_parity.sh           GGUF rows matching the M5 table (results/2026-10-14_f_parity_gguf/parity_table.md)
run_quant.sh            unsloth GGUF quant sweep, GPU + CPU (results/2026-10-14_g_quant_sweep)
run_quant_repeat.sh     repeat GPU session per quant + CPU UD-Q2_K_XL (same folder)
energy_bench/           quant ladder, thinking budget, Thai vs. English benchmark
run_temp0.sh            M5-table rows + every quant at temperature 0, the M5 setting (results/2026-10-14_h_temp0)
modelfiles_quant/       CPU-only (num_gpu 0) Modelfiles for each unsloth quant
```
