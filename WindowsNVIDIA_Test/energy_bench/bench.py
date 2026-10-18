"""Energy benchmark for local SLM inference (Ollama GGUF / OpenAI-compatible MLX servers).

    python bench.py --config configs/ollama_qwen35_4b.yaml --out results/ollama_qwen35_4b
    python bench.py --config configs/mlx_qwen35_4b.yaml    --out results/mlx_qwen35_4b

Resumable: finished requests are skipped (keyed by quant/mode/lang/task/item/repeat), so the
run can be interrupted and restarted with the same command.

Per quant: download (outside measurement) -> load + warm-up -> for each mode: cool-down,
idle baseline, then every request is measured with hardware energy counters read at its
start/end, and split into prefill / thinking / answer windows from the power trace.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import platform
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml


HERE = Path(__file__).resolve().parent


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def http_json(url, body=None, timeout=900):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data, {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def http_stream(url, body, timeout=900):
    """Yield (perf_counter_time, line_bytes) for each streamed line."""
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.strip()
            if line:
                yield time.perf_counter(), line


# ------------------------------------------------------------------------------ backends


class OllamaBackend:
    def __init__(self, cfg):
        self.url = cfg["base_url"].rstrip("/")
        self.num_ctx = cfg.get("num_ctx")
        self.keep_alive = cfg.get("keep_alive", "30m")

    def version(self):
        return http_json(f"{self.url}/api/version").get("version")

    def ensure_model(self, model):
        names = {m["name"] for m in http_json(f"{self.url}/api/tags")["models"]}
        if model in names:
            return
        log(f"pulling {model} ...")
        last = -10
        for _, line in http_stream(f"{self.url}/api/pull", {"model": model}, timeout=7200):
            ev = json.loads(line)
            if "error" in ev:
                raise RuntimeError(ev["error"])
            if ev.get("total") and ev.get("completed"):
                pct = 100 * ev["completed"] / ev["total"]
                if pct - last >= 10:
                    log(f"  {pct:.0f}%"); last = pct
        log("pull done")

    def model_size(self, model):
        return next((m["size"] for m in http_json(f"{self.url}/api/tags")["models"] if m["name"] == model), None)

    def unload_all(self):
        for m in http_json(f"{self.url}/api/ps").get("models", []):
            http_json(f"{self.url}/api/generate", {"model": m["name"], "keep_alive": 0})

    def chat(self, model, messages, think, num_predict, options, seed):
        body = {"model": model, "messages": messages, "stream": True, "think": think,
                "keep_alive": self.keep_alive,
                "options": {**options, "seed": seed, "num_predict": num_predict,
                            **({"num_ctx": self.num_ctx} if self.num_ctx else {})}}
        out = {"content": "", "thinking": "", "t_first": None, "t_first_answer": None}
        final = {}
        for t, line in http_stream(f"{self.url}/api/chat", body):
            ev = json.loads(line)
            if "error" in ev:
                raise RuntimeError(ev["error"])
            msg = ev.get("message", {})
            if msg.get("thinking"):
                out["thinking"] += msg["thinking"]
                out["t_first"] = out["t_first"] or t
            if msg.get("content"):
                out["content"] += msg["content"]
                out["t_first"] = out["t_first"] or t
                out["t_first_answer"] = out["t_first_answer"] or t
            if ev.get("done"):
                final = ev
        ns = 1e9
        out.update(
            in_tokens=final.get("prompt_eval_count"), out_tokens=final.get("eval_count"),
            done_reason=final.get("done_reason"),
            load_s=final.get("load_duration", 0) / ns, prefill_s=final.get("prompt_eval_duration", 0) / ns,
            decode_s=final.get("eval_duration", 0) / ns, server_total_s=final.get("total_duration", 0) / ns)
        return out

    # ------------------------------------------------------------ thinking budget

    def _generate(self, model, prompt, num_predict, options, seed, stop=None):
        """Raw (template-free) streamed generation. Returns (text, t_first, final_event)."""
        body = {"model": model, "prompt": prompt, "raw": True, "stream": True, "keep_alive": self.keep_alive,
                "options": {**options, "seed": seed, "num_predict": num_predict,
                            **({"num_ctx": self.num_ctx} if self.num_ctx else {}),
                            **({"stop": stop} if stop else {})}}
        text, t_first, final = "", None, {}
        for t, line in http_stream(f"{self.url}/api/generate", body):
            ev = json.loads(line)
            if "error" in ev:
                raise RuntimeError(ev["error"])
            if ev.get("response"):
                text += ev["response"]
                t_first = t_first or t
            if ev.get("done"):
                final = ev
        return text, t_first, final

    def chat_budget(self, model, messages, budget, answer_max, options, seed):
        """Budget forcing: think for at most ``budget`` tokens, then close the thinking block
        and generate the answer. Phase 2 reuses phase 1's KV cache (same prefix), so the forced
        continuation costs no extra prefill. budget=0 is the template's empty-thinking prompt.
        """
        prompt = render_chatml(messages)
        ns = 1e9
        if budget > 0:
            thinking, t_first, f1 = self._generate(model, prompt + "<think>\n", budget, options, seed,
                                                   stop=["</think>"])
            answer_prompt = prompt + "<think>\n" + thinking.rstrip() + "\n</think>\n\n"
        else:
            thinking, t_first, f1 = "", None, {}
            answer_prompt = prompt + "<think>\n\n</think>\n\n"
        content, t_ans, f2 = self._generate(model, answer_prompt, answer_max, options, seed)
        first_in = f1.get("prompt_eval_count") if f1 else f2.get("prompt_eval_count")
        return {
            "content": content, "thinking": thinking, "t_first": t_first or t_ans, "t_first_answer": t_ans,
            "in_tokens": first_in,
            "out_tokens": f1.get("eval_count", 0) + f2.get("eval_count", 0),
            "think_tokens": f1.get("eval_count", 0), "answer_tokens": f2.get("eval_count", 0),
            "think_done_reason": f1.get("done_reason") if f1 else None,  # "length" = forced stop
            "done_reason": f2.get("done_reason"),
            "load_s": (f1.get("load_duration", 0) + f2.get("load_duration", 0)) / ns,
            "prefill_s": (f1.get("prompt_eval_duration", 0) + f2.get("prompt_eval_duration", 0)) / ns,
            "decode_s": (f1.get("eval_duration", 0) + f2.get("eval_duration", 0)) / ns,
            "server_total_s": (f1.get("total_duration", 0) + f2.get("total_duration", 0)) / ns,
        }


def render_chatml(messages):
    """Qwen ChatML rendering (system/user turns only) up to the assistant header."""
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return out + "<|im_start|>assistant\n"


class OpenAICompatBackend:
    """mlx_lm.server / LM Studio / llama-server style /v1/chat/completions."""

    def __init__(self, cfg):
        self.url = cfg["base_url"].rstrip("/")
        self.thinking_kwarg = cfg.get("thinking_kwarg", "enable_thinking")

    def version(self):
        return None

    def ensure_model(self, model):
        pass  # servers load on demand; pre-download with `huggingface-cli download` / LM Studio

    def model_size(self, model):
        return None

    def unload_all(self):
        pass

    def chat(self, model, messages, think, num_predict, options, seed):
        body = {"model": model, "messages": messages, "stream": True, "max_tokens": num_predict,
                "seed": seed, "stream_options": {"include_usage": True},
                "chat_template_kwargs": {self.thinking_kwarg: bool(think)}, **options}
        out = {"content": "", "thinking": "", "t_first": None, "t_first_answer": None}
        usage, finish = {}, None
        in_think_tag = False
        for t, line in http_stream(f"{self.url}/v1/chat/completions", body):
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            ev = json.loads(payload)
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                finish = ch.get("finish_reason") or finish
                d = ch.get("delta", {})
                reasoning = d.get("reasoning_content") or d.get("reasoning") or ""
                text = d.get("content") or ""
                # Some servers inline <think>...</think> in content instead of a reasoning field.
                if "<think>" in text:
                    in_think_tag, text = True, text.replace("<think>", "")
                if in_think_tag:
                    if "</think>" in text:
                        r, text = text.split("</think>", 1)
                        reasoning += r; in_think_tag = False
                    else:
                        reasoning, text = reasoning + text, ""
                if reasoning:
                    out["thinking"] += reasoning
                    out["t_first"] = out["t_first"] or t
                if text:
                    out["content"] += text
                    out["t_first"] = out["t_first"] or t
                    if text.strip():
                        out["t_first_answer"] = out["t_first_answer"] or t
        out.update(in_tokens=usage.get("prompt_tokens"), out_tokens=usage.get("completion_tokens"),
                   done_reason=finish, load_s=None, prefill_s=None, decode_s=None, server_total_s=None)
        return out


BACKENDS = {"ollama": OllamaBackend, "openai": OpenAICompatBackend}


# ------------------------------------------------------------------------------ workload


def build_requests(prompts, tasks, langs):
    reqs = []
    for task in tasks:
        spec = prompts["tasks"][task]
        for item in spec["items"]:
            for lang in langs:
                fields = {k: v[lang] for k, v in item.items() if isinstance(v, dict) and lang in v}
                reqs.append({"task": task, "item": item["id"], "lang": lang,
                             "messages": [{"role": "system", "content": prompts["system"][lang]},
                                          {"role": "user", "content": spec["template"][lang].format(**fields)}]})
    return reqs


def expand_modes(modes):
    """A mode with ``think_budgets: [..]`` expands into one mode per budget (budget_<N>)."""
    out = {}
    for name, mc in modes.items():
        if "think_budgets" in mc:
            for b in mc["think_budgets"]:
                out[f"budget_{b}"] = {**{k: v for k, v in mc.items() if k != "think_budgets"},
                                      "think": b > 0, "think_budget": b}
        else:
            out[name] = mc
    return out


def key_of(r):
    return (r["quant"], r["mode"], r["lang"], r["task"], r["item"], r["repeat"])


# ------------------------------------------------------------------------------ gpu state


def make_meter(mcfg):
    """Energy meter on Zeus: NVIDIA/AMD GPU, RAPL CPU on Linux, Apple Silicon, Windows CPU via EMI."""
    from zeus_meter import ZeusTimeline
    return ZeusTimeline(period=mcfg["trace_period_s"], gpu_indices=mcfg.get("gpu_indices"), log=log)


def gpu_state():
    try:
        import pynvml
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return {"gpu_temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                "gpu_sm_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                "gpu_mem_used_gb": round(pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**30, 2)}
    except Exception:
        return {}


def system_info(backend):
    info = {"platform": platform.platform(), "python": sys.version.split()[0],
            "processor": platform.processor(), "backend_version": backend.version()}
    try:
        import pynvml
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info.update(gpu=pynvml.nvmlDeviceGetName(h), driver=pynvml.nvmlSystemGetDriverVersion(),
                    gpu_power_limit_w=pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000,
                    gpu_mem_gb=round(pynvml.nvmlDeviceGetMemoryInfo(h).total / 2**30, 1))
    except Exception:
        pass
    try:
        import psutil
        info.update(ram_gb=round(psutil.virtual_memory().total / 2**30, 1), cpu_count=psutil.cpu_count())
    except Exception:
        pass
    if sys.platform == "darwin":
        info["chip"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                      capture_output=True, text=True).stdout.strip()
    return info


# ------------------------------------------------------------------------------ measurement


def total_power(tl, i):
    vals = [v for v in tl.p[i] if v == v]
    return sum(vals) if vals else float("nan")


def cooldown(tl, mcfg):
    """Wait until total power is stable (max-min < settle_w over 5 s) or timeout."""
    t_end = time.perf_counter() + mcfg["cooldown_max_s"]
    while time.perf_counter() < t_end:
        time.sleep(1)
        now = time.perf_counter()
        idx = [i for i, t in enumerate(tl.t) if t >= now - 5]
        if len(idx) >= 5:
            vals = [total_power(tl, i) for i in idx]
            if max(vals) - min(vals) < mcfg["cooldown_settle_w"]:
                return now
    return time.perf_counter()


def measure_idle(tl, seconds):
    a = tl.mark(); time.sleep(seconds); b = tl.mark()
    dur = b.t - a.t
    return {d: e / dur for d, e in tl.counter_energy(a, b).items()}


def run_request(backend, tl, model, req, mode_cfg, seed, idle_w, tail_s=0.25):
    a = tl.mark()
    st0 = gpu_state()
    if "think_budget" in mode_cfg:
        res = backend.chat_budget(model, req["messages"], mode_cfg["think_budget"], mode_cfg["num_predict"],
                                  mode_cfg.get("options", {}), seed)
    else:
        res = backend.chat(model, req["messages"], mode_cfg["think"], mode_cfg["num_predict"],
                           mode_cfg.get("options", {}), seed)
    t_end = time.perf_counter()
    # NVML's energy counter only updates every ~100 ms: wait so the end mark includes the
    # request's last update. This tail is kept as its own phase (mostly idle, removed in net).
    time.sleep(tail_s)
    b = tl.mark()
    st1 = gpu_state()
    dur = b.t - a.t
    energy = tl.counter_energy(a, b)
    trace = tl.trace_energy(a.t, b.t)

    # windows: prefill = send -> first token; thinking = first token -> first answer token
    # (only if thinking happened); answer = rest.
    t_first = res["t_first"] or t_end
    t_ans = res["t_first_answer"] or t_end
    if not res["thinking"]:
        t_ans = t_first
    windows = {"prefill": (a.t, t_first), "think": (t_first, t_ans), "answer": (t_ans, t_end),
               "tail": (t_end, b.t)}

    row = {"model": model, "energy_total_j": {}, "energy_net_j": {}, "energy_trace_j": trace,
           "peak_power_w": tl.peak_power(a.t, b.t), "t0": a.t, "t1": b.t, "wall_s": dur,
           "gen_s": t_end - a.t, "ttft_s": t_first - a.t,
           "ttfa_s": t_ans - a.t if res["t_first_answer"] else None,
           "think_s": t_ans - t_first, "answer_s": t_end - t_ans}
    for d, e in energy.items():
        row["energy_total_j"][d] = e
        row["energy_net_j"][d] = e - idle_w.get(d, 0.0) * dur
    # phase split: integrate the power trace per window, scaled so phases sum to the counter total
    for phase, (t0, t1) in windows.items():
        tr = tl.trace_energy(t0, t1) if t1 > t0 else {d: 0.0 for d in energy}
        row[f"energy_{phase}_j"] = {
            d: (tr[d] * energy[d] / trace[d] if trace.get(d) and trace[d] == trace[d] and trace[d] > 0 else tr[d])
            for d in energy}
    row.update({k: res[k] for k in ("in_tokens", "out_tokens", "done_reason", "load_s", "prefill_s",
                                     "decode_s", "server_total_s")})
    row.update({k: res[k] for k in ("think_tokens", "answer_tokens", "think_done_reason") if k in res})
    row.update(think_chars=len(res["thinking"]), answer_chars=len(res["content"]),
               content=res["content"], thinking=res["thinking"],
               **{f"start_{k}": v for k, v in st0.items()}, **{f"end_{k}": v for k, v in st1.items()})
    return row


def save_trace(tl, path, t0, t1):
    rows = tl.rows(t0, t1)
    if not rows:
        return
    with gzip.open(path, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quants", nargs="*", help="subset of quants to run")
    ap.add_argument("--modes", nargs="*", help="subset of modes to run")
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.chdir(HERE)
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    prompts = yaml.safe_load(open(cfg["prompts"], encoding="utf-8"))
    mcfg = cfg["measurement"]
    quants = args.quants or cfg["quants"]
    modes = {k: v for k, v in expand_modes(cfg["modes"]).items() if not args.modes or k in args.modes}
    repeats = args.repeats or cfg["repeats"]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "traces").mkdir(exist_ok=True)
    results_path = out / "results.jsonl"

    done = set()
    if results_path.exists():
        for line in open(results_path, encoding="utf-8"):
            done.add(key_of(json.loads(line)))

    plan = []
    for q in quants:
        for mode, mc in modes.items():
            reqs = build_requests(prompts, cfg["tasks"], mc["langs"])
            for rep in range(repeats):
                order = reqs[:]
                random.Random(cfg["seed_base"] + rep).shuffle(order)
                for r in order:
                    plan.append({**r, "quant": q, "mode": mode, "repeat": rep, "seed": cfg["seed_base"] + rep})
    todo = [p for p in plan if key_of(p) not in done]
    log(f"plan: {len(plan)} requests, {len(done)} done, {len(todo)} to run")
    if args.dry_run or not todo:
        return

    backend = BACKENDS[cfg["backend"]](cfg)
    tl = make_meter(mcfg).start()
    json.dump({**system_info(backend), "config": cfg, "energy_domains": tl.domains,
               "meter": mcfg.get("meter", "zeus")},
              open(out / "system.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    with open(results_path, "a", encoding="utf-8") as fout:
        for q in quants:
            q_todo = [p for p in todo if p["quant"] == q]
            if not q_todo:
                continue
            model = cfg["model_template"].format(quant=q)
            backend.ensure_model(model)
            model_bytes = backend.model_size(model)
            backend.unload_all()
            time.sleep(3)
            t = time.perf_counter()
            backend.chat(model, [{"role": "user", "content": "สวัสดี"}], False, 8, {}, 0)  # load
            load_s = time.perf_counter() - t
            for mc in modes.values():  # warm-up each mode once (not measured)
                backend.chat(model, [{"role": "user", "content": "1+1 เท่ากับเท่าไร"}], mc["think"], 256,
                             mc.get("options", {}), 0)
            log(f"[{q}] loaded in {load_s:.1f}s, {gpu_state()}")

            for mode, mc in modes.items():
                m_todo = [p for p in q_todo if p["mode"] == mode]
                if not m_todo:
                    continue
                tl.clear_trace()
                cooldown(tl, mcfg)
                idle_w = measure_idle(tl, mcfg["idle_measure_s"])
                log(f"[{q}/{mode}] idle {idle_w}, {len(m_todo)} requests")
                t_block = time.perf_counter()
                m_block0 = tl.mark()
                for i, p in enumerate(m_todo, 1):
                    row = run_request(backend, tl, model, p, mc, p["seed"], idle_w, mcfg.get("tail_s", 0.25))
                    row.update({k: p[k] for k in ("quant", "mode", "lang", "task", "item", "repeat", "seed")},
                               think=mc["think"], think_budget=mc.get("think_budget"), num_predict=mc["num_predict"], idle_w=idle_w,
                               model_load_s=load_s, ts=time.time())
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n"); fout.flush()
                    gpu = next(iter(row["energy_net_j"].values()))
                    if i % 10 == 0 or i == len(m_todo):
                        log(f"[{q}/{mode}] {i}/{len(m_todo)} last: {p['task']}/{p['lang']} "
                            f"{row['out_tokens']} tok {row['wall_s']:.1f}s {gpu:.0f} J {row['done_reason']}")
                m_block1 = tl.mark()
                with open(out / "blocks.jsonl", "a", encoding="utf-8") as fb:
                    fb.write(json.dumps({"quant": q, "mode": mode, "requests": len(m_todo),
                                         "wall_s": m_block1.t - m_block0.t, "idle_w": idle_w,
                                         "energy_j": tl.counter_energy(m_block0, m_block1),
                                         "load_s": load_s, "model_bytes": model_bytes, "ts": time.time(),
                                         **gpu_state()}) + "\n")
                if mcfg.get("save_traces"):
                    save_trace(tl, out / "traces" / f"{q}__{mode}.csv.gz", t_block, time.perf_counter())
    tl.stop()
    log("all done")


if __name__ == "__main__":
    main()
