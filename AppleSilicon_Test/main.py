"""Sustain_SLM - measure the energy / resource cost of running a Foundry Local model.

Pick which hardware to measure (Intel CPU, NVIDIA GPU, Apple Silicon CPU, Apple Silicon GPU),
run inference, and get an .xlsx report (appended on every execution).

Interactive:   python main.py
Scripted:      python main.py --hardware intel_cpu,nvidia_gpu --runs 3 --warmup
Pipeline test: python main.py --dry-run          (no Foundry Local model needed)
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import platform
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import report
from monitors import ALL_MONITORS, MemoryMonitor

# Hardware -> model variant. Edit these when you switch models.
# Use a GPU/NPU variant alias for "gpu" if you want inference to actually run on the GPU.
MODEL_ALIASES = {
    "cpu": "qwen3.5-4b-generic-cpu:3",
    "gpu": "qwen3.5-4b-generic-gpu:4",
}

# Share the Foundry CLI's model cache (`foundry cache location`), so models downloaded with the
# CLI are found here instead of in a separate per-app cache.
FOUNDRY_CACHE_DIR = str(Path.home() / ".foundry" / "cache" / "models")

DEFAULT_MESSAGES: List[Tuple[str, str]] = [
    ("system", "You are a helpful assistant."),
    ("user", "กระบวนการทางวิทยาศาสตร์มีกี่ขั้นตอนอะไรบ้าง?"),
]

THINKING_LEVELS = ["No", "Low", "Medium", "High"]

DEFINITIONS = [
    ("Definition: time to first token", "first streamed token, including reasoning/thinking tokens if the model emits them"),
    ("Definition: usage change", "first sample -> last sample taken during inference (sampled every ~0.5 s)"),
    ("Definition: usage avg/peak", "mean / max of the samples taken during inference"),
    ("Definition: tokens", "number of streamed text chunks (approximation of tokens); tokens/sec = tokens / inference duration"),
    ("Definition: total energy", "CPU energy + GPU energy of the selected hardware; DRAM, GPU SRAM and ANE energy are shown separately and NOT included"),
    ("Definition: net energy vs idle", "energy - (idle average power x run duration), floored at 0; idle measured before the runs"),
    ("Definition: N/A", "value not measurable on this machine (reason in the Notes column / warnings below); never estimated"),
]


# --------------------------------------------------------------------------- helpers
def ask(prompt: str, default: str = "") -> str:
    try:
        ans = input(f"{prompt} [{default}]: " if default else f"{prompt}: ").strip()
    except EOFError:
        ans = ""
    return ans or default


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hardware", help="comma list of: intel_cpu,nvidia_gpu,apple_cpu,apple_gpu or 'all' (skips the menu)")
    p.add_argument("--alias", help="Foundry Local model variant alias (skips the prompt)")
    p.add_argument("--runs", type=int, help="number of measured runs")
    p.add_argument("--warmup", action="store_true", help="do one un-measured warm-up run first")
    p.add_argument("--thinking", type=str.lower, choices=[l.lower() for l in THINKING_LEVELS],
                   help="thinking level: no, low, medium, high (skips the prompt)")
    p.add_argument("--prompt", help="single user prompt (replaces the default conversation)")
    p.add_argument("--idle-seconds", type=float, default=5.0, help="idle baseline length, 0 to skip")
    p.add_argument("--interval", type=float, default=0.5, help="sampling interval in seconds")
    p.add_argument("--output", default=str(Path(__file__).with_name("sustain_slm_report.xlsx")))
    p.add_argument("--dry-run", action="store_true", help="simulate inference (test the measurement pipeline)")
    return p.parse_args()


def choose_hardware(args: argparse.Namespace, interval: float):
    """Detect hardware, show the menu, return (selected monitor instances, availability list)."""
    detected = []
    for cls in ALL_MONITORS:
        ok, info = cls.detect()
        detected.append((cls, ok, info))

    print("\nDetected hardware:")
    for i, (cls, ok, info) in enumerate(detected, 1):
        status = info if ok else f"unavailable - {info}"
        print(f"  [{i}] {cls.label:<20} {status}")
    available = [d for d in detected if d[1]]

    if not available:
        print("\nNo measurable hardware found: only duration and memory will be recorded.")
        return [], detected

    if args.hardware:
        wanted = [w.strip() for w in args.hardware.split(",") if w.strip()]
        chosen = [d for d in available if "all" in wanted or d[0].key in wanted]
    else:
        raw = ask("\nSelect hardware to measure (numbers separated by commas, or 'all')", "all")
        if raw.lower() == "all":
            chosen = available
        else:
            idx = {int(t) for t in raw.replace(" ", "").split(",") if t.isdigit()}
            chosen = [d for n, d in enumerate(detected, 1) if n in idx and d[1]]
    if not chosen:
        sys.exit("No valid hardware selected.")

    monitors, seen_kinds = [], set()
    for cls, _, _ in chosen:
        if cls.kind in seen_kinds:
            print(f"  ! Skipping {cls.label}: a {cls.kind.upper()} is already selected "
                  f"(the report has one CPU block and one GPU block)")
            continue
        seen_kinds.add(cls.kind)
        monitors.append(cls(interval=interval))
    print("Selected: " + ", ".join(m.label for m in monitors))
    return monitors, detected


def choose_thinking(args: argparse.Namespace) -> str:
    """Ask for the thinking level (No / Low / Medium / High). Returns the capitalised name."""
    if args.thinking:
        return args.thinking.capitalize()
    print("\nThinking level:")
    for i, lvl in enumerate(THINKING_LEVELS, 1):
        print(f"  [{i}] {lvl}")
    raw = ask("Select thinking level (number or name)", "1").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(THINKING_LEVELS):
        return THINKING_LEVELS[int(raw) - 1]
    for lvl in THINKING_LEVELS:
        if raw.lower() == lvl.lower():
            return lvl
    print("  ! Not recognised, using 'No'")
    return "No"


def apply_thinking_prompt(messages, level: str):
    """Foundry Local: the SDK exposes no thinking switch, so the level is requested through the prompt.

    'Reasoning: <level>' in the system message is the convention used by gpt-oss style models and
    '/no_think' is the Qwen3 soft switch. How much effect it has depends on the model.
    """
    msgs = [list(m) for m in messages]
    hint = "Reasoning: none. Do not think step by step; answer directly." if level == "No" \
        else f"Reasoning: {level.lower()}"
    if msgs and msgs[0][0] == "system":
        msgs[0][1] = f"{msgs[0][1]}\n{hint}"
    else:
        msgs.insert(0, ["system", hint])
    if level == "No":
        for m in reversed(msgs):
            if m[0] == "user":
                m[1] = f"{m[1]} /no_think"
                break
    method = ("Prompt-level hint ('Reasoning: ...'" + (" + '/no_think'" if level == "No" else "")
              + "); Foundry Local SDK has no native thinking control, effect depends on the model")
    return [tuple(m) for m in msgs], method


def list_variants() -> List[Any]:
    """Downloaded chat model variants (same as `foundry cache list`); empty list if none or on error."""
    try:
        variants = [v for v in foundry_manager().catalog.get_cached_models() if "chat" in (v.info.task or "")]
    except Exception as e:
        print(f"  ! Could not list downloaded Foundry Local models: {e}")
        return []
    return sorted(variants, key=lambda v: (v.alias, v.id))


def pick_alias(args: argparse.Namespace, monitors) -> str:
    if args.alias:
        return args.alias
    default = MODEL_ALIASES["gpu"] if any(m.kind == "gpu" for m in monitors) else MODEL_ALIASES["cpu"]
    variants = list_variants()
    if not variants:
        print(f"  ! No downloaded models found in {FOUNDRY_CACHE_DIR} (download one with `foundry model download <alias>`)")
        return ask("Model alias", default)

    print("\nDownloaded models:")
    for i, v in enumerate(variants, 1):
        rt = v.info.runtime
        device = rt.device_type.value if rt else "?"
        size = f"{v.info.file_size_mb / 1024:.1f} GB" if v.info.file_size_mb else "?"
        print(f"  [{i}] {v.id:<40} {device:<4} {size:>8}")
    ids = [v.id for v in variants]
    default_choice = str(ids.index(default) + 1) if default in ids else default
    raw = ask("Select model (number or alias)", default_choice).strip()
    if raw.isdigit() and 1 <= int(raw) <= len(ids):
        return ids[int(raw) - 1]
    return raw


# --------------------------------------------------------------------------- inference
def foundry_manager():
    """Initialise the Foundry Local manager once and return the singleton."""
    from foundry_local_sdk import Configuration, FoundryLocalManager

    if FoundryLocalManager.instance is None:
        FoundryLocalManager.initialize(Configuration(app_name="Sustain_SLM", model_cache_dir=FOUNDRY_CACHE_DIR))
    return FoundryLocalManager.instance


def foundry_load(alias: str):
    manager = foundry_manager()
    # GPU variants (e.g. WebGPU) need their execution provider registered before load().
    # Already-registered EPs are skipped; only missing ones are downloaded.
    if any(not ep.is_registered for ep in manager.discover_eps()):
        print("Registering execution providers ...")
        result = manager.download_and_register_eps()
        if not result.success:
            print(f"  ! Execution provider registration incomplete: {result.status} (failed: {result.failed_eps})")
    model = manager.catalog.get_model_variant(alias)
    if model is None:
        raise ValueError(f"Foundry Local model variant not found: {alias}")
    model.download()
    model.load()
    return manager, model


def foundry_infer(model, messages, echo: bool) -> Dict[str, Any]:
    from foundry_local_sdk import ChatSession, MessageItem, Request, TextItem

    text, chunks, ttft = [], 0, None
    with ChatSession(model) as session:
        session.set_streaming(True)
        with Request() as request:
            for role, content in messages:
                request.add_item(getattr(MessageItem, role)(content))
            t_req = time.perf_counter()
            with session.process_streaming_request(request) as stream:
                for item in stream:
                    if isinstance(item, TextItem):
                        if ttft is None:
                            ttft = time.perf_counter() - t_req
                        chunks += 1
                        text.append(item.text)
                        if echo:
                            print(item.text, end="", flush=True)
    if echo:
        print()
    return {"text": "".join(text), "tokens": chunks, "ttft_s": ttft}


def dry_infer(model, messages, echo: bool) -> Dict[str, Any]:
    """Fake workload: CPU burn + memory ballast, to exercise the monitors and the report."""
    ballast = bytearray(b"\x01") * (64 << 20)
    t_req = time.perf_counter()
    time.sleep(0.3)
    ttft = time.perf_counter() - t_req
    chunks = 0
    for _ in range(20):
        end = time.perf_counter() + 0.08
        while time.perf_counter() < end:
            pass
        time.sleep(0.04)
        chunks += 1
    return {"text": f"[dry run, {len(ballast) >> 20} MB ballast]", "tokens": chunks, "ttft_s": ttft}


def measured_run(infer: Callable, model, messages, monitors, mem, echo: bool, n: int) -> Dict[str, Any]:
    mem.start()
    for m in monitors:
        m.start()
    t0 = time.perf_counter()
    try:
        out = infer(model, messages, echo)
    finally:
        duration = time.perf_counter() - t0
        for m in monitors:
            m.stop()
        mem.stop()
    return {
        "run": n, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s": duration, "ttft_s": out["ttft_s"], "tokens": out["tokens"], "text": out["text"],
        "exact_tokens": out.get("exact_tokens", False),
        "reasoning_chunks": out.get("reasoning_chunks", 0),
        "results": {m.key: m.result() for m in monitors}, "memory": mem.result(),
    }


def idle_baseline(monitors, seconds: float) -> Dict[str, Dict[str, Any]]:
    idle: Dict[str, Dict[str, Any]] = {}
    if seconds <= 0 or not monitors:
        return idle
    print(f"\nMeasuring idle baseline for {seconds:.0f}s - please leave the machine idle...")
    for m in monitors:
        m.start()
    time.sleep(seconds)
    for m in monitors:
        m.stop()
        r = m.result()
        idle[m.key] = {"power_w": r["power_avg_w"], "usage_avg": r["usage_avg"], "notes": r["notes"]}
        p = r["power_avg_w"]
        print(f"  {m.label}: idle power {'N/A' if p is None else f'{p:.2f} W'}")
    return idle


# --------------------------------------------------------------------------- environment
def collect_env(selected, detected, idle, args, alias, session=None) -> List[Tuple[str, Any]]:
    items: List[Tuple[str, Any]] = [
        ("OS", platform.platform()), ("Machine", platform.machine()),
        ("Python", sys.version.split()[0]), ("Model alias", alias),
        ("Sampling interval (s)", args.interval),
        ("Idle baseline (s)", args.idle_seconds),
        ("Dry run", args.dry_run),
    ]
    if session:
        items += [("Thinking level", session["thinking"]), ("Thinking method", session["thinking_method"])]
    for pkg in ("psutil", "openpyxl", "py-cpuinfo", "nvidia-ml-py", "zeus-apple-silicon", "foundry-local-sdk"):
        try:
            items.append((f"Package: {pkg}", md.version(pkg)))
        except md.PackageNotFoundError:
            items.append((f"Package: {pkg}", "not installed"))
    for cls, ok, info in detected:
        items.append((f"Detected: {cls.label}", info if ok else f"unavailable - {info}"))
    for m in selected:
        items.append((f"Selected: {m.label}", m.hardware_name()))
        backend = {"intel_cpu": "psutil + Intel RAPL (/sys/class/powercap)",
                   "nvidia_gpu": "NVML (nvidia-ml-py) energy counter",
                   "apple_cpu": "psutil + zeus-apple-silicon (IOReport)",
                   "apple_gpu": "ioreg + zeus-apple-silicon (IOReport)"}[m.key]
        items.append((f"Backend: {m.label}", backend))
        ip = idle.get(m.key, {}).get("power_w")
        items.append((f"Idle power: {m.label} (W)", "N/A" if ip is None else round(ip, 3)))
    items += DEFINITIONS
    return items


def print_run(row: Dict[str, Any]) -> None:
    keys = ["Thinking level", "Inference duration (s)", "Time to first token (s)", "Tokens generated (~stream chunks)",
            "Tokens/sec", "Process RAM peak (MB)", "CPU hardware", "CPU usage change (% -> %)",
            "CPU power avg (W)", "CPU energy (J)", "GPU hardware", "GPU usage change (% -> %)",
            "GPU power avg (W)", "GPU energy (J)", "Total energy (J)", "Total energy (Wh)",
            "Avg total power (W)", "Energy per token (J/token)"]
    print(f"\n--- Run {row['Run #']} ---")
    for k in keys:
        print(f"  {k:<36} {row[k]}")
    if row["Notes"]:
        print(f"  Notes: {row['Notes']}")


# --------------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    monitors, detected = choose_hardware(args, args.interval)
    thinking = choose_thinking(args)
    alias = "dry-run" if args.dry_run else pick_alias(args, monitors)
    n_runs = args.runs or int(ask("Number of measured runs", "1") or 1)
    warmup = args.warmup or (not args.hardware and ask("Warm-up run first? (y/n)", "n").lower().startswith("y"))
    messages = [("user", args.prompt)] if args.prompt else DEFAULT_MESSAGES
    prompt_text = [c for r, c in messages if r == "user"][-1]
    messages, think_method = apply_thinking_prompt(messages, thinking)
    if args.dry_run:
        think_method = "dry run: not applied"

    manager = model = None
    infer = dry_infer
    if not args.dry_run:
        print(f"\nLoading model {alias} ...")
        manager, model = foundry_load(alias)
        infer = foundry_infer

    mem = MemoryMonitor(interval=0.25)
    session = {
        "id": uuid.uuid4().hex[:8], "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "alias": alias, "prompt": prompt_text,
        "thinking": thinking, "thinking_method": think_method,
        "selected": [{"key": m.key, "label": m.label, "kind": m.kind} for m in monitors],
    }
    try:
        if warmup:
            print("\nWarm-up run (not recorded)...")
            infer(model, messages, True)
        session["idle"] = idle_baseline(monitors, args.idle_seconds)

        runs = []
        for n in range(1, n_runs + 1):
            print(f"\nRun {n}/{n_runs}: measuring...")
            run = measured_run(infer, model, messages, monitors, mem, True, n)
            runs.append(run)
            print_run(report.build_row(session, run))
    finally:
        if model is not None:
            model.unload()
        if manager is not None:
            manager.close()

    env = collect_env(monitors, detected, session["idle"], args, alias, session)
    for cls, ok, info in detected:
        if not ok:
            env.append(("Warning", f"{cls.label} unavailable: {info}"))
    saved = report.write_report(Path(args.output), session, runs, env)
    print(f"\nReport saved to: {saved}")


if __name__ == "__main__":
    main()
