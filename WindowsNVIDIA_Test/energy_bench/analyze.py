"""Summaries, automatic scoring and figures for a bench.py run.

    python analyze.py results/ollama_qwen35_4b [--prompts configs/prompts.yaml]

Writes <run>/analysis/: summary CSVs, report.md and figures (PNG).

Energy aggregation: per-request NVML energy is quantized to the counter's ~100 ms update
interval, so short requests are noisy individually. Ratios such as J/token are therefore
computed as sum(energy) / sum(tokens) over all requests of a condition, which averages the
quantization error out, rather than as a mean of per-request ratios.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
QUANT_ORDER = ["BF16", "bf16", "F16", "Q8_0", "8bit", "Q6_K", "6bit", "Q5_K_M", "5bit", "Q4_K_M", "4bit",
               "Q3_K_M", "3bit", "UD-Q2_K_XL", "mixed_2_6"]
MODE_LABEL = {"no_think": "Thinking off", "think": "Thinking on"}
PHASES = ["prefill", "think", "answer", "tail"]

# reference palette (dataviz skill): categorical slots 1-4, chrome & ink
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MODE_COLOR = {"no_think": S1, "think": S2}
LANG_COLOR = {"th": S1, "en": S2}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.family": ["Segoe UI", "Tahoma", "DejaVu Sans"], "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 8, "figure.dpi": 150,
})

THAI = re.compile(r"[฀-๿]")
LATIN = re.compile(r"[A-Za-z]")
CJK = re.compile(r"[぀-ヿ一-鿿가-힯]")


# ------------------------------------------------------------------------------ loading


def load(run: Path, prompts_path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in open(run / "results.jsonl", encoding="utf-8")]
    df = pd.json_normalize(rows, sep=".")
    domains = [c[len("energy_total_j."):] for c in df.columns if c.startswith("energy_total_j.")]
    df["e_total"] = df[[f"energy_total_j.{d}" for d in domains]].sum(axis=1)
    df["e_net"] = df[[f"energy_net_j.{d}" for d in domains]].sum(axis=1)
    for ph in PHASES:
        cols = [f"energy_{ph}_j.{d}" for d in domains if f"energy_{ph}_j.{d}" in df]
        df[f"e_{ph}"] = df[cols].sum(axis=1) if cols else 0.0
    df["peak_w"] = df[[c for c in df.columns if c.startswith("peak_power_w.")]].max(axis=1)
    df["decode_tps"] = df["out_tokens"] / df["decode_s"].where(df["decode_s"] > 0)
    fallback = df["out_tokens"] / (df["gen_s"] - df["ttft_s"])
    df["decode_tps"] = df["decode_tps"].fillna(fallback)
    df["finished"] = (df["done_reason"] == "stop") & (df["content"].str.strip().str.len() > 0)
    df["cjk_mix"] = df["content"].map(lambda s: bool(CJK.search(s or "")))
    df["thai_share"] = df["content"].map(_thai_share)
    df["correct"] = _score(df, yaml.safe_load(open(prompts_path, encoding="utf-8")))
    order = [q for q in QUANT_ORDER if q in set(df["quant"])] + sorted(set(df["quant"]) - set(QUANT_ORDER))
    df["quant"] = pd.Categorical(df["quant"], categories=order, ordered=True)
    df.attrs["domains"] = domains
    return df


def _thai_share(s):
    th, la = len(THAI.findall(s or "")), len(LATIN.findall(s or ""))
    return th / (th + la) if th + la else float("nan")


def _score(df, prompts):
    gold = {}
    for task in ("qa", "classify"):
        for it in prompts["tasks"][task]["items"]:
            gold[it["id"]] = it
    labels = prompts["tasks"]["classify"]["labels"]

    def score(r):
        it = gold.get(r["item"])
        if it is None:
            return float("nan")  # summarize / humor: needs human or LLM-judge evaluation
        text = (r["content"] or "").strip()
        if r["task"] == "qa":
            return float(any(g.lower() in text.lower() for g in it["gold_any"]))
        # classify: earliest label mentioned wins
        hits = [(text.lower().find(k.lower()), v) for k, v in labels[r["lang"]].items() if k.lower() in text.lower()]
        return float(min(hits)[1] == it["gold"]) if hits else 0.0

    return df.apply(score, axis=1)


# ------------------------------------------------------------------------------ tables


def agg(g: pd.DataFrame) -> pd.Series:
    n = len(g)
    return pd.Series({
        "n": n,
        "finish_rate": g["finished"].mean(),
        "accuracy": g["correct"].mean(),
        "in_tokens_mean": g["in_tokens"].mean(),
        "out_tokens_mean": g["out_tokens"].mean(),
        "out_tokens_sd": g["out_tokens"].std(),
        "j_per_req_mean": g["e_net"].mean(),
        "j_per_req_sd": g["e_net"].std(),
        "j_per_req_ci95": 1.96 * g["e_net"].std() / n ** 0.5,
        "j_per_token": g["e_net"].sum() / g["out_tokens"].sum(),
        "j_total_per_token": g["e_total"].sum() / g["out_tokens"].sum(),
        "wall_s_mean": g["wall_s"].mean(),
        "ttft_ms_median": 1000 * g["ttft_s"].median(),
        "decode_tps_median": g["decode_tps"].median(),
        "peak_w_median": g["peak_w"].median(),
        "cjk_mix_rate": g["cjk_mix"].mean(),  # Chinese/Japanese/Korean leaking into the answer
        "wh_per_1k_req": g["e_net"].mean() * 1000 / 3600,
    })


def write_tables(df, out: Path, blocks: pd.DataFrame | None):
    by_cond = df.groupby(["quant", "mode", "lang"], observed=True).apply(agg, include_groups=False).reset_index()
    by_task = df.groupby(["quant", "mode", "lang", "task"], observed=True).apply(agg, include_groups=False).reset_index()
    by_cond.to_csv(out / "summary_by_condition.csv", index=False)
    by_task.to_csv(out / "summary_by_task.csv", index=False)

    # Thai vs English on identical items (thinking off): tokens and energy ratios
    nt = df[df["mode"] == "no_think"]
    if set(nt["lang"]) >= {"th", "en"}:
        p = nt.groupby(["quant", "task", "item", "lang"], observed=True)[["in_tokens", "out_tokens", "e_net"]].sum()
        p = p.unstack("lang")
        ratio = pd.DataFrame({
            "in_tokens_th": p[("in_tokens", "th")], "in_tokens_en": p[("in_tokens", "en")],
            "in_ratio_th_en": p[("in_tokens", "th")] / p[("in_tokens", "en")],
            "out_ratio_th_en": p[("out_tokens", "th")] / p[("out_tokens", "en")],
            "energy_ratio_th_en": p[("e_net", "th")] / p[("e_net", "en")],
        }).reset_index()
        ratio.to_csv(out / "th_vs_en.csv", index=False)
    else:
        ratio = None

    # phase split (share of total energy)
    ph = df.groupby(["quant", "mode"], observed=True)[[f"e_{p}" for p in PHASES] + ["e_total"]].sum()
    for p in PHASES:
        ph[f"share_{p}"] = ph[f"e_{p}"] / ph["e_total"]
    ph.reset_index().to_csv(out / "phase_split.csv", index=False)

    # cross-check: per-request counter sums vs whole-block counter
    check = None
    if blocks is not None and len(blocks):
        req = df.groupby(["quant", "mode"], observed=True)["e_total"].sum().rename("sum_requests_j")
        blk = blocks.groupby(["quant", "mode"])["energy_j_sum"].sum().rename("block_counter_j")
        check = pd.concat([req, blk], axis=1).dropna()
        check["requests_share"] = check["sum_requests_j"] / check["block_counter_j"]
        check.reset_index().to_csv(out / "block_crosscheck.csv", index=False)
    return by_cond, by_task, ratio, ph, check


# ------------------------------------------------------------------------------ figures


def _finish(fig, path, note=None, top=1.0):
    if note:
        fig.text(0.01, 0.005, note, color=MUTED, fontsize=8, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03 if note else 0, 1, top))
    fig.savefig(path)
    plt.close(fig)


def fig_j_per_token(by_cond, out):
    d = by_cond[by_cond["lang"] == "th"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for mode, g in d.groupby("mode", observed=True):
        g = g.sort_values("quant")
        ax.plot(g["quant"].astype(str), g["j_per_token"], marker="o", color=MODE_COLOR[mode], label=MODE_LABEL[mode])
        last = g.iloc[-1]
        ax.annotate(MODE_LABEL[mode], (len(g) - 1, last["j_per_token"]), xytext=(6, 0),
                    textcoords="offset points", color=INK2, va="center", fontsize=9)
    ax.set_title("Net GPU energy per generated token (Thai prompts)")
    ax.set_ylabel("J / output token")
    ax.set_xlabel("Quantization (largest → smallest)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    _finish(fig, out / "fig1_j_per_token.png", "Net = measured energy minus idle baseline. sum(J)/sum(tokens) per condition.")


def fig_j_per_request(by_cond, out):
    d = by_cond[by_cond["lang"] == "th"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for mode, g in d.groupby("mode", observed=True):
        g = g.sort_values("quant")
        ax.errorbar(g["quant"].astype(str), g["j_per_req_mean"], yerr=g["j_per_req_ci95"], marker="o",
                    color=MODE_COLOR[mode], label=MODE_LABEL[mode], capsize=3, elinewidth=1)
    ax.set_yscale("log")
    ax.set_title("Net GPU energy per request (Thai prompts)")
    ax.set_ylabel("J / request (log scale, mean, 95% CI)")
    ax.set_xlabel("Quantization (largest → smallest)")
    ax.legend(loc="center right")
    _finish(fig, out / "fig2_j_per_request.png")


def fig_accuracy_vs_energy(by_cond, by_task, out):
    scored = by_task[by_task["task"].isin(["qa", "classify"]) & (by_task["lang"] == "th")]
    if scored.empty:
        return
    acc = scored.groupby(["quant", "mode"], observed=True).apply(
        lambda g: (g["accuracy"] * g["n"]).sum() / g["n"].sum(), include_groups=False).rename("acc").reset_index()
    e = by_cond[by_cond["lang"] == "th"][["quant", "mode", "j_per_req_mean"]]
    d = acc.merge(e, on=["quant", "mode"])
    modes = [m for m in ("no_think", "think") if m in set(d["mode"])]
    fig, axes = plt.subplots(1, len(modes), figsize=(5.2 * len(modes), 4.3), sharey=True, squeeze=False)
    for ax, mode in zip(axes[0], modes):
        g = d[d["mode"] == mode].sort_values("acc")
        ax.scatter(g["j_per_req_mean"], g["acc"] * 100, s=64, color=MODE_COLOR[mode],
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        # points that share an accuracy level: alternate labels above / below, left to right
        for _, grp in g.groupby(g["acc"].round(2)):
            for i, (_, r) in enumerate(grp.sort_values("j_per_req_mean").iterrows()):
                above = i % 2 == 0
                ax.annotate(str(r["quant"]), (r["j_per_req_mean"], r["acc"] * 100),
                            xytext=(0, 9 if above else -10), textcoords="offset points", fontsize=8,
                            color=INK2, ha="center", va="bottom" if above else "top")
        lo, hi = g["j_per_req_mean"].min(), g["j_per_req_mean"].max()
        pad = (hi - lo) * 0.25 or hi * 0.1
        ax.set_xlim(lo - pad, hi + pad * 1.6)
        ax.set_title(MODE_LABEL[mode])
        ax.set_xlabel("Net J / request (mean, all Thai tasks)")
    axes[0][0].set_ylabel("Accuracy (%)")
    axes[0][0].set_ylim(50, 95)
    fig.suptitle("Accuracy vs. energy (Thai QA + sentiment)", x=0.01, ha="left", fontweight="bold", fontsize=12)
    _finish(fig, out / "fig3_accuracy_vs_energy.png",
            "Accuracy is automatic: QA = gold answer present; sentiment = exact label. Note the different x-scales.")


def fig_th_vs_en(df, out):
    nt = df[df["mode"] == "no_think"]
    if set(nt["lang"]) < {"th", "en"}:
        return
    tasks = ["qa", "summarize", "classify", "humor"]
    g = nt.groupby(["task", "lang"])
    tok = g["in_tokens"].mean().unstack("lang").reindex(tasks)
    eng = g["e_net"].mean().unstack("lang").reindex(tasks)
    outtok = g["out_tokens"].mean().unstack("lang").reindex(tasks)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    for ax, data, title, ylab in [
        (axes[0], tok, "Prompt tokens", "tokens (mean)"),
        (axes[1], outtok, "Output tokens", "tokens (mean)"),
        (axes[2], eng, "Net energy per request", "J (mean)"),
    ]:
        x = range(len(tasks)); w = 0.38
        for k, lang in enumerate(["th", "en"]):
            ax.bar([i + (k - 0.5) * w for i in x], data[lang], width=w - 0.02, color=LANG_COLOR[lang],
                   label={"th": "Thai", "en": "English"}[lang])
        ax.set_xticks(list(x), tasks)
        ax.set_title(title); ax.set_ylabel(ylab); ax.grid(axis="x", visible=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncols=2)
    _finish(fig, out / "fig4_thai_vs_english.png",
            "Thinking off; semantically equivalent Thai/English prompts; all quantizations pooled.", top=0.9)


def fig_finish_rate(by_task, out):
    d = by_task[(by_task["mode"] == "think") & (by_task["lang"] == "th")]
    if d.empty:
        return
    colors = {"qa": S1, "summarize": S2, "classify": S3, "humor": S4}
    fig, ax = plt.subplots(figsize=(7, 4))
    for task in ["qa", "summarize", "classify", "humor"]:
        g = d[d["task"] == task].sort_values("quant")
        if g.empty:
            continue
        ax.plot(g["quant"].astype(str), g["finish_rate"] * 100, marker="o", color=colors[task], label=task)
    ax.set_title("Thinking runs that finished within the token cap")
    ax.set_ylabel("Finished (%)")
    ax.set_xlabel("Quantization (largest → smallest)")
    ax.set_ylim(-5, 105)
    ax.legend(loc="lower center", ncols=4, bbox_to_anchor=(0.5, 1.02))
    ax.set_title("Thinking runs that finished within the token cap", pad=28)
    _finish(fig, out / "fig5_think_finish_rate.png")


def fig_phase_split(ph, out):
    d = ph.reset_index()
    d = d[d["mode"] == "think"].sort_values("quant")
    if d.empty:
        return
    colors = {"prefill": S1, "think": S2, "answer": S3, "tail": S4}
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = [0.0] * len(d)
    xs = d["quant"].astype(str).tolist()
    for p in PHASES:
        vals = (d[f"share_{p}"] * 100).tolist()
        ax.bar(xs, vals, bottom=bottom, color=colors[p], label=p, edgecolor=SURFACE, linewidth=2, width=0.6)
        for x, b, v in zip(xs, bottom, vals):
            if p == "think":
                ax.annotate(f"{v:.0f}%", (x, b + v / 2), ha="center", va="center", color=SURFACE, fontsize=9,
                            fontweight="bold")
        bottom = [b + v for b, v in zip(bottom, vals)]
    ax.set_title("Where the energy goes with thinking on")
    ax.set_xlabel("prefill / answer / tail shares are listed in phase_split.csv")
    ax.set_ylabel("Share of request energy (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper center", ncols=4, bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="x", visible=False)
    _finish(fig, out / "fig6_phase_split.png")


def fig_speed(by_cond, blocks, out):
    d = by_cond[(by_cond["lang"] == "th") & (by_cond["mode"] == "no_think")].sort_values("quant")
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(d["quant"].astype(str), d["decode_tps_median"], marker="o", color=S1)
    axes[0].set_title("Decode speed"); axes[0].set_ylabel("tokens / s (median)"); axes[0].set_ylim(bottom=0)
    if blocks is not None and "model_bytes" in blocks and blocks["model_bytes"].notna().any():
        sz = blocks.groupby("quant")["model_bytes"].first() / 1e9
        sz = sz.reindex([q for q in d["quant"].astype(str) if q in sz.index])
        axes[1].bar(sz.index, sz.values, color=S1, width=0.6)
        axes[1].set_title("Model file size"); axes[1].set_ylabel("GB"); axes[1].grid(axis="x", visible=False)
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
    _finish(fig, out / "fig7_speed_and_size.png")


# ------------------------------------------------------------------------------ report


def fmt_table(df: pd.DataFrame, cols: dict) -> str:
    head = "| " + " | ".join(cols.values()) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append("–" if v != v else (f"{v:.0%}" if c in ("finish_rate", "accuracy", "cjk_mix_rate", "forced_rate", "answered_rate")
                                                 else f"{v:,.3f}" if abs(v) < 10 else f"{v:,.0f}"))
            else:
                cells.append(str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def write_report(run, df, by_cond, ratio, check, out):
    sysinfo = json.load(open(run / "system.json", encoding="utf-8")) if (run / "system.json").exists() else {}
    lines = [f"# Energy benchmark report: `{run.name}`", ""]
    if sysinfo:
        lines += [f"- Hardware: {sysinfo.get('gpu') or sysinfo.get('chip', '?')}, "
                  f"{sysinfo.get('ram_gb', '?')} GB RAM; {sysinfo.get('platform', '')}",
                  f"- Backend: {sysinfo.get('config', {}).get('backend')} {sysinfo.get('backend_version') or ''}; "
                  f"energy domains: {', '.join(sysinfo.get('energy_domains', []))}", ""]
    lines += [f"Requests: {len(df)}; finished: {df['finished'].mean():.0%}.", "",
              "## Per condition", "",
              fmt_table(by_cond, {"quant": "Quant", "mode": "Mode", "lang": "Lang", "n": "n",
                                  "finish_rate": "Finished", "accuracy": "Acc (QA+cls)",
                                  "out_tokens_mean": "Out tok", "j_per_req_mean": "J/req",
                                  "j_per_token": "J/tok (net)", "decode_tps_median": "tok/s",
                                  "ttft_ms_median": "TTFT ms", "peak_w_median": "Peak W",
                                  "cjk_mix_rate": "CJK mix"})]
    if ratio is not None:
        r = ratio.groupby("task")[["in_ratio_th_en", "out_ratio_th_en", "energy_ratio_th_en"]].median()
        lines += ["## Thai vs English (thinking off, median ratio th/en per item)", "",
                  fmt_table(r.reset_index(), {"task": "Task", "in_ratio_th_en": "Prompt tokens",
                                              "out_ratio_th_en": "Output tokens",
                                              "energy_ratio_th_en": "Energy"})]
    if check is not None and len(check):
        lines += ["## Measurement cross-check", "",
                  "Sum of per-request counter energy vs. one counter reading over the whole block "
                  "(the remainder is idle time between requests).", "",
                  fmt_table(check.reset_index(), {"quant": "Quant", "mode": "Mode", "sum_requests_j": "Σ requests J",
                                                  "block_counter_j": "Block J", "requests_share": "Share"})]
    lines += ["## Figures", ""] + [f"![{p.stem}]({p.name})" for p in sorted(out.glob("fig*.png"))]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------ thinking budget

QUANT_COLOR_SEQ = [S1, S2, S3, S4]


def budget_tables(df, out):
    d = df[df["think_budget"].notna()].copy()
    d["think_budget"] = d["think_budget"].astype(int)
    d["forced"] = d["think_done_reason"] == "length"
    d["_task"] = d["task"]  # kept inside groups even when grouping by task

    def agg_b(g):
        sc = g[g["_task"].isin(["qa", "classify"])]
        n_sc = sc["correct"].notna().sum()
        acc = sc["correct"].mean()
        return pd.Series({
            "n": len(g), "accuracy": acc, "accuracy_n": n_sc,
            "accuracy_ci95": 1.96 * (acc * (1 - acc) / n_sc) ** 0.5 if n_sc else float("nan"),
            "think_tokens_mean": g["think_tokens"].mean(), "answer_tokens_mean": g["answer_tokens"].mean(),
            "forced_rate": g["forced"].mean(), "answered_rate": g["finished"].mean(),
            "j_per_req_mean": g["e_net"].mean(), "j_per_req_ci95": 1.96 * g["e_net"].std() / len(g) ** 0.5,
            "j_per_correct": g.loc[g["_task"].isin(["qa", "classify"]), "e_net"].sum() / max(sc["correct"].sum(), 1),
            "wall_s_mean": g["wall_s"].mean(),
        })

    by_b = d.groupby(["quant", "think_budget"], observed=True).apply(agg_b, include_groups=False).reset_index()
    by_bt = d.groupby(["quant", "think_budget", "task"], observed=True).apply(agg_b, include_groups=False).reset_index()
    by_b.to_csv(out / "budget_summary.csv", index=False)
    by_bt.to_csv(out / "budget_by_task.csv", index=False)
    return by_b, by_bt


def _quant_colors(by_b):
    return {q: QUANT_COLOR_SEQ[i % 4] for i, q in enumerate(by_b["quant"].astype(str).unique())}


def fig_budget_energy_accuracy(by_b, out):
    colors = _quant_colors(by_b)
    budgets = sorted(by_b["think_budget"].unique())
    xs = {b: i for i, b in enumerate(budgets)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for q, g in by_b.groupby("quant", observed=True):
        g = g.sort_values("think_budget"); x = [xs[b] for b in g["think_budget"]]
        axes[0].errorbar(x, g["j_per_req_mean"], yerr=g["j_per_req_ci95"], marker="o", capsize=3, elinewidth=1,
                         color=colors[str(q)], label=str(q))
        axes[1].errorbar(x, g["accuracy"] * 100, yerr=g["accuracy_ci95"] * 100, marker="o", capsize=3,
                         elinewidth=1, color=colors[str(q)], label=str(q))
    for ax in axes:
        ax.set_xticks(range(len(budgets)), [str(b) for b in budgets])
        ax.set_xlabel("Thinking budget (tokens)")
    axes[0].set_yscale("log")
    axes[0].set_title("Net GPU energy per request"); axes[0].set_ylabel("J (log scale, mean, 95% CI)")
    axes[1].set_title("Accuracy (Thai QA + sentiment)"); axes[1].set_ylabel("Accuracy (%) ± 95% CI")
    axes[1].set_ylim(40, 100)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncols=len(colors))
    _finish(fig, out / "fig8_budget_energy_accuracy.png",
            "Budget forcing: thinking is cut at the budget, closed with </think>, then the answer is generated. "
            "0 = no thinking.", top=0.9)


def fig_budget_thinking_use(by_b, by_bt, out):
    colors = _quant_colors(by_b)
    budgets = sorted(by_b["think_budget"].unique())
    xs = {b: i for i, b in enumerate(budgets)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for q, g in by_b.groupby("quant", observed=True):
        g = g.sort_values("think_budget")
        axes[0].plot([xs[b] for b in g["think_budget"]], g["think_tokens_mean"], marker="o",
                     color=colors[str(q)], label=str(q))
    axes[0].plot(range(len(budgets)), budgets, linestyle="--", linewidth=1, color=MUTED, label="budget (cap)")
    axes[0].set_yscale("symlog", linthresh=100)
    axes[0].set_title("Thinking tokens actually used"); axes[0].set_ylabel("tokens (mean, symlog)")
    first_q = str(by_b["quant"].astype(str).iloc[0])
    tcol = {"qa": S1, "summarize": S2, "classify": S3, "humor": S4}
    d = by_bt[(by_bt["quant"].astype(str) == first_q) & (by_bt["think_budget"] > 0)]
    for task in ["qa", "summarize", "classify", "humor"]:
        g = d[d["task"] == task].sort_values("think_budget")
        axes[1].plot([xs[b] for b in g["think_budget"]], g["forced_rate"] * 100, marker="o", color=tcol[task],
                     label=task)
    axes[1].set_title(f"Thinking cut off by the budget ({first_q})"); axes[1].set_ylabel("Forced stop (%)")
    axes[1].set_ylim(-5, 105)
    for ax in axes:
        ax.set_xticks(range(len(budgets)), [str(b) for b in budgets]); ax.set_xlabel("Thinking budget (tokens)")
        ax.legend(loc="lower left" if ax is axes[1] else "upper left", fontsize=8)
    _finish(fig, out / "fig9_budget_thinking_use.png")


def fig_budget_task_accuracy(by_bt, out):
    budgets = sorted(by_bt["think_budget"].unique())
    xs = {b: i for i, b in enumerate(budgets)}
    quants = by_bt["quant"].astype(str).unique()
    fig, axes = plt.subplots(1, len(quants), figsize=(5.5 * len(quants), 4), sharey=True, squeeze=False)
    for ax, q in zip(axes[0], quants):
        d = by_bt[by_bt["quant"].astype(str) == q]
        for task, color in (("qa", S1), ("classify", S3)):
            g = d[d["task"] == task].sort_values("think_budget")
            ax.plot([xs[b] for b in g["think_budget"]], g["accuracy"] * 100, marker="o", color=color, label=task)
        ax.set_title(q); ax.set_xticks(range(len(budgets)), [str(b) for b in budgets])
        ax.set_xlabel("Thinking budget (tokens)")
    axes[0][0].set_ylabel("Accuracy (%)"); axes[0][0].set_ylim(0, 105); axes[0][0].legend(loc="lower right")
    fig.suptitle("Accuracy per task vs. thinking budget", x=0.01, ha="left", fontweight="bold", fontsize=12)
    _finish(fig, out / "fig10_budget_task_accuracy.png", "n = 25 per point (5 items × 5 repeats).")


def write_budget_report(run, df, by_b, out):
    lines = [f"# Thinking-budget report: `{run.name}`", "",
             f"Requests: {len(df)}. Budget forcing on Thai prompts; accuracy = QA + sentiment (n=50 per row).", "",
             fmt_table(by_b, {"quant": "Quant", "think_budget": "Budget", "n": "n", "accuracy": "Acc",
                              "think_tokens_mean": "Think tok", "forced_rate": "Forced",
                              "answered_rate": "Answered", "j_per_req_mean": "J/req",
                              "j_per_correct": "J per correct", "wall_s_mean": "Wall s"}),
             "## Figures", ""] + [f"![{p.stem}]({p.name})" for p in sorted(out.glob("fig*.png"))]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--prompts", default=str(HERE / "configs" / "prompts.yaml"))
    args = ap.parse_args()
    run = Path(args.run)
    out = run / "analysis"; out.mkdir(exist_ok=True)
    df = load(run, Path(args.prompts))
    blocks = None
    if (run / "blocks.jsonl").exists():
        blocks = pd.json_normalize([json.loads(l) for l in open(run / "blocks.jsonl", encoding="utf-8")], sep=".")
        blocks["energy_j_sum"] = blocks[[c for c in blocks.columns if c.startswith("energy_j.")]].sum(axis=1)
    by_cond, by_task, ratio, ph, check = write_tables(df, out, blocks)
    if "think_budget" in df and df["think_budget"].notna().any():
        by_b, by_bt = budget_tables(df, out)
        fig_budget_energy_accuracy(by_b, out)
        fig_budget_thinking_use(by_b, by_bt, out)
        fig_budget_task_accuracy(by_bt, out)
        write_budget_report(run, df, by_b, out)
        print(f"wrote {out}")
        return
    fig_j_per_token(by_cond, out)
    fig_j_per_request(by_cond, out)
    fig_accuracy_vs_energy(by_cond, by_task, out)
    fig_th_vs_en(df, out)
    fig_finish_rate(by_task, out)
    fig_phase_split(ph, out)
    fig_speed(by_cond, blocks, out)
    write_report(run, df, by_cond, ratio, check, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
