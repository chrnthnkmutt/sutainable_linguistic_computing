# Sustainable Linguistic Computing

Tools and experiment results for measuring the resource use and energy consumption of small language models (SLMs).

This repository supports reproducible inference benchmarks across different models, hardware platforms, languages, and reasoning settings. Measurements are collected during model runs and exported to Excel reports for comparison.

> The repository name is retained as `sutainable_linguistic_computing` for compatibility.

## What this project measures

Depending on the platform and available hardware, each run can record:

- Inference duration and time to first token
- Generated tokens and throughput
- CPU and GPU utilisation
- CPU and GPU power and energy
- Net energy relative to an idle baseline
- Process memory and GPU memory
- Model placement and reasoning activity
- Environment details, warnings, and measurement limitations

Results are written to `.xlsx` files containing summary rows, time-series samples, run statistics, and environment metadata.

## Repository layout

| Directory | Purpose |
|---|---|
| [`AppleSilicon_Test/`](AppleSilicon_Test/) | Measurements on Apple Silicon Macs using Foundry Local or Ollama |
| [`WindowsNVIDIA_Test/`](WindowsNVIDIA_Test/) | Windows and NVIDIA measurements, including benchmark scripts and prompt sets |
| `main.py` | Foundry Local measurement entry point |
| `main_ollama.py` | Ollama measurement entry point |
| `report.py` | Excel report generation |
| `monitors/` | CPU, GPU, memory, power, and energy monitors |
| `prompts/` | Thai and English benchmark prompts and reference answers |
| `energy_bench/` | Quantisation, reasoning-budget, and language-comparison benchmarks |
| `summarize.py` | Converts reports into summary tables |

The platform-specific directories contain the most complete setup instructions for their respective environments.

## Supported runtimes

### Foundry Local

`main.py` uses the Foundry Local SDK to serve a model and collect measurements.

### Ollama

`main_ollama.py` uses Ollama's OpenAI-compatible API. It supports local or remote Ollama endpoints and can measure both CPU-only and GPU-backed model variants.

## Hardware support

| Hardware | Utilisation | Power / energy measurement |
|---|---|---|
| Intel or AMD CPU | `psutil` | Linux RAPL or Windows Energy Meter where available |
| NVIDIA GPU | NVML | NVML energy counter |
| Apple Silicon CPU | `psutil` | IOReport through `zeus-apple-silicon` |
| Apple Silicon GPU | `ioreg` | IOReport through `zeus-apple-silicon` |

Unavailable hardware is reported as unavailable rather than estimated. Selecting a device determines what is measured; it does not force the model to execute on that device. For meaningful comparisons, use a model or runtime configuration that places inference on the intended CPU or GPU.

## Installation

From the platform-specific test directory:

```bash
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Install and prepare the runtime you intend to use:

```bash
# Ollama
ollama serve
ollama pull qwen3:1.7b
```

For Foundry Local, keep the existing `foundry_local_sdk` installation available in the environment.

## Running an experiment

### Interactive run with Ollama

```bash
python main_ollama.py
```

### Scripted run

```bash
python main_ollama.py \
  --model qwen3:1.7b \
  --hardware nvidia_gpu,intel_cpu \
  --runs 3 \
  --warmup \
  --thinking low
```

### Apple Silicon example

```bash
python main_ollama.py \
  --model gpt-oss:20b \
  --hardware apple_cpu,apple_gpu \
  --runs 3 \
  --warmup
```

### Foundry Local example

```bash
python main.py \
  --hardware intel_cpu,nvidia_gpu \
  --runs 3 \
  --warmup \
  --thinking no
```

### Test the measurement pipeline without a model

```bash
python main_ollama.py --dry-run
# or
python main.py --dry-run
```

Reports are saved as `sustain_slm_report_ollama.xlsx` or `sustain_slm_report.xlsx` unless another output path is supplied.

## Common options

Both entry points support the following options:

| Option | Description |
|---|---|
| `--hardware LIST` | Comma-separated hardware selection: `intel_cpu`, `nvidia_gpu`, `apple_cpu`, `apple_gpu`, or `all` |
| `--runs N` | Number of measured runs |
| `--warmup` | Run one unmeasured request before collecting results |
| `--thinking LEVEL` | Reasoning level: `no`, `low`, `medium`, or `high` |
| `--prompt TEXT` | Use a single custom prompt |
| `--interval S` | Sampling interval for utilisation and power |
| `--idle-seconds S` | Duration of the idle baseline |
| `--output FILE` | Excel report destination |
| `--dry-run` | Exercise the pipeline without loading a model |

Ollama-specific options include `--model`, `--base-url`, `--max-tokens`, `--temperature`, and, in the Windows benchmark version, `--prompt-file`, `--settle-max`, and `--tail-seconds`.

See the detailed guides for all options and platform-specific considerations:

- [Apple Silicon setup and usage](AppleSilicon_Test/README.md)
- [Windows/NVIDIA setup and benchmarks](WindowsNVIDIA_Test/README.md)

## Reports and interpretation

Each report can contain the following worksheets:

- **Summary** — one row per measured run
- **Samples** — time-series measurements for charting
- **Statistics** — mean and standard deviation for the latest session
- **Environment** — operating system, library versions, hardware, measurement backends, and warnings

Important measurement conventions:

- Energy is reported only when a supported hardware counter is available.
- `N/A` values are not estimates; the reason is recorded in the run notes.
- Net energy is calculated relative to the measured idle baseline.
- Token counts may differ by runtime: Foundry Local counts streamed chunks, while Ollama uses completion-token metadata.
- Time to first token includes reasoning tokens when the model emits them.
- A warm-up run is recommended because loading a model can significantly increase the first request's time and energy.

## Experimental methodology

For more comparable results:

1. Close unnecessary applications and keep the machine idle before and between runs.
2. Use the same model, prompt, reasoning level, token limit, and sampling settings.
3. Run a warm-up request before measured runs.
4. Collect multiple runs and inspect the Statistics worksheet.
5. Record the operating system, power mode, model quantisation, and model placement.
6. For remote Ollama, remember that the monitors measure the machine running the benchmark script, not the remote inference server.

## Benchmark data

The Windows/NVIDIA experiment directory includes additional workloads for:

- Thai and English prompt comparisons
- Quantisation sweeps
- Reasoning-budget comparisons
- CPU-only versus GPU-backed execution
- Accuracy summaries for selected question-answering and sentiment prompts

Generated results are generally kept out of version control. See the platform-specific README files for benchmark commands and result locations.

## License and status

This repository is an experimental research project. Measurement accuracy depends on the operating system, hardware counters, drivers, runtime configuration, model placement, and background system activity. Treat results as comparative measurements under the documented conditions rather than universal power ratings.
