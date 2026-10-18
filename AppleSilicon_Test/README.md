# Sustain_SLM

Measures the resource and energy cost of running a small language model and writes the results to an Excel report.

| Entry point | Model served by | Calls it via |
|---|---|---|
| `main.py` | Foundry Local | `foundry_local_sdk` |
| `main_ollama.py` | Ollama | OpenAI-compatible API (`http://localhost:11434/v1`) |

Both use the same hardware monitors (`monitors/`) and the same report writer (`report.py`).

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
| Intel CPU | psutil | Intel RAPL, **Linux only**, needs `sudo` | Intel CPU |
| NVIDIA GPU | NVML | NVML energy counter | NVIDIA driver, `nvidia-ml-py` |
| Apple Silicon CPU | psutil | IOReport via `zeus-apple-silicon` (no sudo) | Apple Silicon Mac |
| Apple Silicon GPU | `ioreg` | IOReport via `zeus-apple-silicon` (no sudo) | Apple Silicon Mac |

- Hardware that is not present is shown as unavailable, with the reason, and cannot be selected.
- You can select one CPU and one GPU per session (the report has one CPU block and one GPU block).
- On Windows, Intel CPU shows usage only; power and energy are `N/A`.
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
| `--idle-seconds S` | 5 | Idle baseline length before the runs. `0` skips it (net-energy columns become `N/A`). |
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
| `--base-url URL` | `http://localhost:11434/v1` | OpenAI-compatible endpoint. |
| `--api-key KEY` | `ollama` | Ignored by Ollama, required by the client. |
| `--max-tokens N` | unlimited | Cap generated tokens, useful for comparing runs fairly. |
| `--temperature T` | 0 | Sampling temperature. |

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
- **Net energy vs idle:** energy minus (idle average power x run duration), floored at 0.
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
main_pyjoules_backup.py your original script (PyJoules version)
```
