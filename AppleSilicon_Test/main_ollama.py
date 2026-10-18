"""Sustain_SLM (Ollama edition) - same measurements and xlsx report as main.py, but the model is
served by Ollama and called through its OpenAI-compatible API (http://localhost:11434/v1).

Interactive:   python main_ollama.py
Scripted:      python main_ollama.py --model qwen3:1.7b --hardware nvidia_gpu,intel_cpu --runs 3
Pipeline test: python main_ollama.py --dry-run

Notes
  * Ollama runs as a separate server process: its RAM is added to "Process RAM".
  * The first request makes Ollama load the model, which inflates time/energy -> use the warm-up run.
  * Tokens are the exact completion_tokens reported by Ollama (falls back to stream chunks).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import main as base
import report
from monitors import MemoryMonitor

DEFAULT_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hardware", help="comma list of: intel_cpu,nvidia_gpu,apple_cpu,apple_gpu or 'all'")
    p.add_argument("--model", help="Ollama model tag, e.g. qwen3:1.7b (skips the prompt)")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"OpenAI-compatible endpoint (default {DEFAULT_BASE_URL})")
    p.add_argument("--api-key", default="ollama", help="ignored by Ollama, required by the client")
    p.add_argument("--runs", type=int, help="number of measured runs")
    p.add_argument("--warmup", action="store_true", help="un-measured warm-up run first (recommended)")
    p.add_argument("--thinking", type=str.lower, choices=[l.lower() for l in base.THINKING_LEVELS],
                   help="thinking level: no, low, medium, high (skips the prompt)")
    p.add_argument("--prompt", help="single user prompt (replaces the default conversation)")
    p.add_argument("--max-tokens", type=int, default=None, help="cap generated tokens (fairer run-to-run comparison)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--idle-seconds", type=float, default=5.0)
    p.add_argument("--interval", type=float, default=0.5)
    p.add_argument("--output", default=str(Path(__file__).with_name("sustain_slm_report_ollama.xlsx")))
    p.add_argument("--dry-run", action="store_true", help="simulate inference (no Ollama needed)")
    return p.parse_args()


def connect(args: argparse.Namespace):
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("The 'openai' package is required: pip install openai")
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    try:
        models = [m.id for m in client.models.list().data]
    except Exception as exc:
        sys.exit(f"Cannot reach Ollama at {args.base_url} ({exc}).\nIs it running? Try: ollama serve")
    return client, models


def pick_model(args: argparse.Namespace, models) -> str:
    if args.model:
        if models and args.model not in models:
            print(f"  ! '{args.model}' is not in Ollama's local list (ollama pull {args.model}); trying anyway")
        return args.model
    if not models:
        sys.exit("Ollama has no local models. Pull one first, e.g.: ollama pull qwen3:1.7b")
    print("\nModels available in Ollama:")
    for i, m in enumerate(models, 1):
        print(f"  [{i}] {m}")
    raw = base.ask("Select model (number or tag)", "1")
    return models[int(raw) - 1] if raw.isdigit() and 1 <= int(raw) <= len(models) else raw


EFFORT = {"No": "none", "Low": "low", "Medium": "medium", "High": "high"}


def thinking_support(args: argparse.Namespace, model_tag: str, level: str):
    """Ask Ollama's native /api/show what thinking values the model accepts. Returns warning list."""
    import json
    import urllib.request

    origin = args.base_url.rstrip("/")
    origin = origin[:-3] if origin.endswith("/v1") else origin
    try:
        req = urllib.request.Request(origin + "/api/show", data=json.dumps({"model": model_tag}).encode(),
                                     headers={"Content-Type": "application/json"})
        info = json.load(urllib.request.urlopen(req, timeout=10))
    except Exception as exc:
        return [f"Could not read model thinking capabilities ({exc}); request sent anyway"]
    values = (info.get("thinking") or {}).get("values") if isinstance(info.get("thinking"), dict) else None
    caps = info.get("capabilities") or []
    if values is None:
        if "thinking" not in caps:
            return ["Model does not report thinking support: the requested level is probably ignored"]
        return ["Model reports thinking support without named levels: Low/Medium/High are probably all 'on'"] \
            if level != "No" else []
    if values == [False] or values == []:
        return ["Model does not support thinking: the requested level is ignored"]
    if all(isinstance(v, bool) for v in values):
        return ["Model has on/off thinking only: Low, Medium and High are all treated as 'on'"] if level != "No" else []
    named = [str(v).lower() for v in values]
    if level != "No" and level.lower() not in named:
        return [f"Level '{level}' not supported by this model (supports {named}): Ollama uses the model default"]
    if level == "No":
        return ["This model has named thinking levels only; 'No' may not fully disable thinking"] if "none" not in named else []
    return []


def make_infer(client, model_tag: str, args: argparse.Namespace):
    def infer(_model, messages, echo: bool) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(
            model=model_tag,
            messages=[{"role": r, "content": c} for r, c in messages],
            stream=True, temperature=args.temperature,
            stream_options={"include_usage": True},
            extra_body={"reasoning_effort": EFFORT[args.thinking_level]},
        )
        if args.max_tokens:
            kwargs["max_tokens"] = args.max_tokens
        text, chunks, ttft, usage_tokens, reasoning_chunks = [], 0, None, None, 0
        t_req = time.perf_counter()
        for chunk in client.chat.completions.create(**kwargs):
            usage = getattr(chunk, "usage", None)
            if usage is not None and getattr(usage, "completion_tokens", None) is not None:
                usage_tokens = usage.completion_tokens
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            piece = getattr(delta, "content", None) or ""
            thinking = getattr(delta, "reasoning", None) or getattr(delta, "reasoning_content", None) or ""
            if thinking:
                reasoning_chunks += 1
            if piece or thinking:
                if ttft is None:
                    ttft = time.perf_counter() - t_req
                chunks += 1
                text.append(piece)
                if echo and piece:
                    print(piece, end="", flush=True)
        if echo:
            print()
        return {"text": "".join(text), "tokens": usage_tokens if usage_tokens is not None else chunks,
                "ttft_s": ttft, "exact_tokens": usage_tokens is not None,
                "reasoning_chunks": reasoning_chunks}
    return infer


def main() -> None:
    args = parse_args()
    monitors, detected = base.choose_hardware(args, args.interval)
    args.thinking_level = base.choose_thinking(args)

    client = None
    if args.dry_run:
        model_tag, infer = "dry-run", base.dry_infer
    else:
        client, models = connect(args)
        model_tag = pick_model(args, models)
        infer = make_infer(client, model_tag, args)
    n_runs = args.runs or int(base.ask("Number of measured runs", "1") or 1)
    warmup = args.warmup or (not args.hardware and base.ask("Warm-up run first? (recommended: loads the model)", "y").lower().startswith("y"))
    messages = [("user", args.prompt)] if args.prompt else base.DEFAULT_MESSAGES
    prompt_text = [c for r, c in messages if r == "user"][-1]

    think_warnings = [] if args.dry_run else thinking_support(args, model_tag, args.thinking_level)
    for w in think_warnings:
        print(f"  ! {w}")
    mem = MemoryMonitor(interval=0.25, watch_names=("ollama",))
    session = {
        "id": uuid.uuid4().hex[:8], "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "alias": f"ollama:{model_tag}", "prompt": prompt_text,
        "thinking": args.thinking_level,
        "thinking_method": "dry run: not applied" if args.dry_run
        else f"Ollama OpenAI API reasoning_effort={EFFORT[args.thinking_level]}",
        "selected": [{"key": m.key, "label": m.label, "kind": m.kind} for m in monitors],
    }
    if warmup:
        print("\nWarm-up run (not recorded)...")
        infer(None, messages, True)
    session["idle"] = base.idle_baseline(monitors, args.idle_seconds)

    runs = []
    for n in range(1, n_runs + 1):
        print(f"\nRun {n}/{n_runs}: measuring...")
        run = base.measured_run(infer, None, messages, monitors, mem, True, n)
        run["notes"] = ["Process RAM includes the Ollama server/runner processes"] + think_warnings
        run["notes"].append(f"Reasoning chunks streamed: {run.get('reasoning_chunks', 0)}")
        run["notes"].append("Tokens = completion_tokens reported by Ollama" if run_exact(run) else
                            "Tokens = stream chunks (Ollama did not report usage)")
        runs.append(run)
        base.print_run(report.build_row(session, run))

    env = base.collect_env(monitors, detected, session["idle"], args, session["alias"], session)
    env += [("Backend: inference", "Ollama via OpenAI-compatible API"), ("Ollama endpoint", args.base_url),
            ("Temperature", args.temperature), ("Max tokens", args.max_tokens or "unlimited"),
            ("Package: openai", _ver("openai"))]
    for cls, ok, info in detected:
        if not ok:
            env.append(("Warning", f"{cls.label} unavailable: {info}"))
    print(f"\nReport saved to: {report.write_report(Path(args.output), session, runs, env)}")


def run_exact(run: Dict[str, Any]) -> bool:
    return bool(run.get("exact_tokens"))


def _ver(pkg: str) -> str:
    import importlib.metadata as md
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return "not installed"


if __name__ == "__main__":
    main()
