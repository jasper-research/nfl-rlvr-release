"""Deterministic build of the paper's headline figures and tables.

Single source of truth: reads the per-game prediction files (full test 2024, n=5185)
plus the empirical-rate target from data/grpo/test.jsonl, computes every reported number
with the SAME metric code used in training/eval (reward/metrics.py), and emits:

    results.json          locked numbers (every cell of every table/figure)
    figs/F3_reliability   reliability diagrams (base, WS1, WS2, DeepSeek, Vegas)
    figs/F4_triangulation Brier dot plot with bootstrap CIs + info-ceiling band
    tables/T1_main.tex     main results table
    tables/T3_decoupling.tex  decoupling comparison + paired bootstrap

All models are scored on the SAME 5185 test plays in the SAME order (verified by
matching outcome vectors). Bootstrap is nonparametric over plays, seeded, reproducible.

Run:  python paper/build_assets.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent          # paper2_fanhuddle/
RES = ROOT / "eval" / "results"
PAPER = Path(__file__).resolve().parent
TABLES = PAPER / "tables"
TABLES.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT / "reward"))               # import metrics without package juggling
import metrics as M  # noqa: E402
sys.path.insert(0, str(PAPER))
import figstyle as fs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

N_BINS = 10
BOOT_BRIER = 10000
BOOT_ECE = 5000
SEED = 0


# --------------------------------------------------------------------------- data
def _load(path: Path):
    d = json.load(open(path))
    if isinstance(d, list):
        return {"_": d}
    return d


def _col(recs, key):
    return np.array([r[key] for r in recs], dtype=np.float64)


def load_models():
    ws1 = _load(RES / "preds_test_ws1-pb.json")          # zeroshot (direct base) + checkpoint-200
    ws2 = _load(RES / "preds_test_ws2-pb.json")          # zeroshot (CoT base)   + checkpoint-200
    ds = _load(RES / "preds_test_deepseek-chat.json")    # single key
    ds_recs = next(iter(ds.values()))

    grpo = [json.loads(l) for l in open(ROOT / "data" / "grpo" / "test.jsonl")]
    teacher = np.array([r["target"] for r in grpo], dtype=np.float64)
    y_grpo = np.array([r["actual_outcome"] for r in grpo], dtype=np.float64)

    y = _col(ws1["checkpoint-200"], "outcome")
    vegas = _col(ws1["checkpoint-200"], "vegas")

    # Every source must describe the same plays in the same order.
    for name, arr in [("ws2", _col(ws2["checkpoint-200"], "outcome")),
                      ("deepseek", _col(ds_recs, "outcome")),
                      ("grpo", y_grpo)]:
        if not np.array_equal(arr, y):
            raise SystemExit(f"outcome vector mismatch: {name} not aligned with ws1")

    models = {
        "base_cot":    _col(ws2["zeroshot"], "prob"),
        "base_direct": _col(ws1["zeroshot"], "prob"),
        "ws1":         _col(ws1["checkpoint-200"], "prob"),
        "ws2":         _col(ws2["checkpoint-200"], "prob"),
        "deepseek":    _col(ds_recs, "prob"),
        "teacher":     teacher,
        "vegas":       vegas,
    }
    models = {k: np.clip(v, 0.0, 1.0) for k, v in models.items()}
    return models, y


META = {
    "base_cot":    dict(name=r"Qwen2.5-7B zero-shot (CoT)",   color=fs.NULL,   role="base"),
    "base_direct": dict(name=r"Qwen2.5-7B zero-shot (direct)", color=fs.NULL_L, role="base"),
    "ws1":         dict(name=r"Direct RLVR (ours)",            color=fs.ACCENT, role="ours"),
    "ws2":         dict(name=r"Masked-CoT RLVR (ours)",        color=fs.COOL,   role="ours"),
    "deepseek":    dict(name=r"DeepSeek-V4 zero-shot",         color=fs.GREEN,  role="ref"),
    "teacher":     dict(name=r"Empirical rate $\hat{p}$",      color=fs.BAR,    role="ref"),
    "vegas":       dict(name=r"Betting market",                color=fs.INK,    role="ceiling"),
}


# ---------------------------------------------------------------------- statistics
def metric_row(p, y):
    mu = M.murphy_decomposition(p, y, n_bins=N_BINS)
    return dict(
        brier=M.brier_score(p, y),
        ece=M.expected_calibration_error(p, y, n_bins=N_BINS),
        mce=M.maximum_calibration_error(p, y, n_bins=N_BINS),
        acc=M.binary_accuracy(p, y),
        reliability=mu["reliability"],
        resolution=mu["resolution"],
        uncertainty=mu["uncertainty"],
    )


def boot_ci(stat_fn, p, y, reps, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, n, n)
        vals[i] = stat_fn(p[idx], y[idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def boot_brier_ci(p, y, reps=BOOT_BRIER, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    se = (p - y) ** 2
    vals = np.array([se[rng.integers(0, n, n)].mean() for _ in range(reps)])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_brier(p_a, p_b, y, reps=BOOT_BRIER, seed=SEED):
    """Mean per-play Brier difference (a - b) with 95% CI. Negative => a better."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diff = (p_a - y) ** 2 - (p_b - y) ** 2
    boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(reps)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(delta=float(diff.mean()), lo=float(lo), hi=float(hi),
               sig=bool(lo > 0 or hi < 0))


# --------------------------------------------------------------------------- compute
def compute(models, y):
    rows = {}
    for k, p in models.items():
        r = metric_row(p, y)
        r["brier_ci"] = boot_brier_ci(p, y)
        r["ece_ci"] = boot_ci(lambda a, b: M.expected_calibration_error(a, b, N_BINS),
                              p, y, BOOT_ECE)
        r["coverage"] = float(np.mean((p >= 0) & (p <= 1)))
        rows[k] = r

    pairs = {
        "ws1_vs_ws2":      ("ws1", "ws2"),
        "ws1_vs_vegas":    ("ws1", "vegas"),
        "ws2_vs_vegas":    ("ws2", "vegas"),
        "ws1_vs_deepseek": ("ws1", "deepseek"),
        "deepseek_vs_vegas": ("deepseek", "vegas"),
    }
    paired = {name: paired_brier(models[a], models[b], y) for name, (a, b) in pairs.items()}
    return {"n": int(len(y)), "n_bins": N_BINS, "models": rows, "paired_brier": paired}


# ----------------------------------------------------------------------------- F3
def fig_reliability(models, y, results):
    panels = ["base_cot", "ws1", "ws2", "deepseek", "vegas"]
    fig, axes = plt.subplots(2, 3, figsize=(fs.WIDE, 4.5))
    axes = axes.ravel()
    for ax, k in zip(axes, panels):
        p = models[k]
        conf, acc, cnt = M.reliability_curve(p, y, n_bins=N_BINS)
        valid = cnt > 0
        ax.plot([0, 1], [0, 1], color=fs.NULL, lw=0.8, ls="--", zorder=1)
        sizes = 8 + 90 * (cnt[valid] / cnt.max())
        ax.scatter(conf[valid], acc[valid], s=sizes, color=META[k]["color"],
                   edgecolor="white", lw=0.5, zorder=3)
        ax.plot(conf[valid], acc[valid], color=META[k]["color"], lw=1.0, zorder=2, alpha=0.7)
        r = results["models"][k]
        ax.set_title(META[k]["name"], fontsize=8)
        ax.text(0.04, 0.92, f"Brier {r['brier']:.3f}\nECE {r['ece']:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7, color=fs.INK)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        ax.set_aspect("equal")
    axes[-1].axis("off")
    fig.supxlabel("predicted win probability", fontsize=8.5, y=-0.01)
    fig.supylabel("observed win frequency", fontsize=8.5, x=-0.01)
    fig.tight_layout(w_pad=1.2, h_pad=1.4)
    fs.save(fig, "F3_reliability")


# ----------------------------------------------------------------------------- F4
def fig_triangulation(results):
    order = ["base_direct", "base_cot", "ws2", "ws1", "deepseek", "teacher", "vegas"]
    order = sorted(order, key=lambda k: -results["models"][k]["brier"])
    fig, ax = plt.subplots(figsize=(fs.WIDE, 3.0))

    # static-information ceiling band: teacher, ws1, deepseek cluster
    ceil_vals = [results["models"][k]["brier"] for k in ("teacher", "ws1", "deepseek")]
    ax.axvspan(min(ceil_vals) - 0.0008, max(ceil_vals) + 0.0008, color=fs.CEIL, alpha=0.16, zorder=0)
    vegas_b = results["models"]["vegas"]["brier"]
    ax.axvline(vegas_b, color=fs.INK, lw=0.9, ls="--", zorder=1)

    for i, k in enumerate(order):
        r = results["models"][k]
        lo, hi = r["brier_ci"]
        ax.plot([lo, hi], [i, i], color=META[k]["color"], lw=1.6, zorder=2,
                solid_capstyle="round")
        ax.scatter([r["brier"]], [i], s=42, color=META[k]["color"], edgecolor="white",
                   lw=0.6, zorder=3)
        ax.text(hi + 0.0015, i, f"{r['brier']:.4f}", va="center", ha="left",
                fontsize=7, color=fs.INK)

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([META[k]["name"] for k in order])
    ax.set_xlabel("Brier score on held-out 2024 (lower is better), 95% CI")
    ax.set_ylim(-0.6, len(order) - 0.4)
    fs.halo(ax.text(sum(ceil_vals) / 3, len(order) - 0.5, "static-information ceiling",
                    ha="center", va="bottom", fontsize=7, color="#8a6d1f", style="italic"))
    fs.halo(ax.text(vegas_b - 0.0007, len(order) - 2.0, "market", rotation=90,
                    ha="center", va="center", fontsize=7, color=fs.INK))
    fig.tight_layout()
    fs.save(fig, "F4_triangulation")


# ---------------------------------------------------------------------- LaTeX tables
def _ci(lo, hi, prec=4):
    return rf"{{\scriptsize[{lo:.{prec}f},\,{hi:.{prec}f}]}}"


def table_main(results):
    order = ["base_direct", "base_cot", "ws2", "ws1", "deepseek", "teacher", "vegas"]
    lines = [
        "% requires \\booktabs. Brier shows 95% bootstrap CI (10k reps over plays).",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"System & Brier & ECE & MCE & Acc. & Resolution \\",
        r"\midrule",
    ]
    for k in order:
        r = results["models"][k]
        b = rf"{r['brier']:.4f}\,{_ci(*r['brier_ci'])}"
        lines.append(
            rf"{META[k]['name']} & {b} & {r['ece']:.4f} & {r['mce']:.4f} & "
            rf"{r['acc']:.3f} & {r['resolution']:.4f} \\"
        )
        if k == "deepseek":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "T1_main.tex").write_text("\n".join(lines) + "\n")


FAITH_CELLS = ["base", "v9", "v6", "basemask", "ws2"]
FAITH_NAME = {
    "base": "Base (CoT)",
    "v9": r"Naive CoT-RLVR ($\hat{p}$ reward)",
    "v6": "Naive CoT-RLVR (outcome reward)",
    "basemask": "Base (masked prompt)",
    "ws2": "Masked-CoT RLVR (ours)",
}
FAITH_COLOR = {"base": fs.NULL, "v9": fs.NULL, "v6": fs.NULL, "basemask": fs.BAR, "ws2": fs.COOL}


def compute_faithfulness(reps=2000, seed=SEED):
    recs = [json.loads(l) for l in open(RES / "reasoning_judged_eval.jsonl")]
    rng = np.random.default_rng(seed)
    out = {}
    for lab in FAITH_CELLS:
        flags = []
        for r in recs:
            m = r["models"].get(lab) or {}
            v = m.get("verdict") if isinstance(m, dict) else None
            f = (v or {}).get("faithfulness", {}).get("v") if isinstance(v, dict) else None
            if f in ("consistent", "inconsistent", "unclear"):
                flags.append(1 if f == "inconsistent" else 0)
        flags = np.array(flags, dtype=float)
        n = len(flags)
        boot = np.array([flags[rng.integers(0, n, n)].mean() for _ in range(reps)])
        out[lab] = dict(rate=float(flags.mean()), lo=float(np.percentile(boot, 2.5)),
                        hi=float(np.percentile(boot, 97.5)), n=int(n))
    return out


def fig_faithfulness(faith):
    fig, ax = plt.subplots(figsize=(fs.WIDE, 2.7))
    for i, lab in enumerate(FAITH_CELLS):
        d = faith[lab]
        r, lo, hi = d["rate"] * 100, d["lo"] * 100, d["hi"] * 100
        ax.barh(i, r, color=FAITH_COLOR[lab], height=0.62, zorder=2)
        ax.plot([lo, hi], [i, i], color=fs.INK, lw=1.1, zorder=3, solid_capstyle="round")
        ax.text(hi + 0.6, i, f"{r:.1f}%", va="center", ha="left", fontsize=7.5, color=fs.INK)
    ax.set_yticks(range(len(FAITH_CELLS)))
    ax.set_yticklabels([FAITH_NAME[l] for l in FAITH_CELLS])
    ax.invert_yaxis()
    ax.set_xlabel("inconsistent reasoning: stated probability does not follow from the CoT (%), "
                  "95% CI")
    ax.set_xlim(0, max(faith[l]["hi"] for l in FAITH_CELLS) * 100 + 6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fs.save(fig, "F5_faithfulness")


def fig_trajectory():
    """F2: in-training held-out calibration vs step. Coupled CoT-RL (v9) decalibrates; isolating
    the gradient (masked-CoT, or direct) calibrates. v9 from W&B; masked/direct from train logs.
    Base anchor (step 0, CoT prompt): Brier 0.2525 / ECE 0.1917, shared by the CoT-prompt runs."""
    intr = json.load(open(PAPER / "figdata" / "intraining_eval.json"))["runs"]
    wb = json.load(open(PAPER / "figdata" / "wandb_eval.json"))["runs"]
    base_b, base_e = 0.2525, 0.1917

    def line(traj, metric, with_base):
        items = sorted((int(k), v) for k, v in traj.items())
        xs = [k for k, _ in items]
        ys = [v[metric] for _, v in items]
        if with_base and xs and xs[0] != 0:
            xs = [0] + xs
            ys = [(base_b if metric == "brier" else base_e)] + ys
        return xs, ys

    specs = [  # label, traj, color, with_base (CoT-prompt runs share the base anchor)
        ("Coupled CoT-RLVR",          wb["v9_cot_phat"]["traj"], fs.INK,    True),
        ("Masked-CoT RLVR (ours)",    intr["masked_lead"],       fs.COOL,   True),
        ("Direct RLVR (ours)",        intr["direct_lead"],       fs.ACCENT, False),
    ]
    fig, (axB, axE) = plt.subplots(1, 2, figsize=(fs.WIDE, 2.8))
    for label, traj, c, wb_base in specs:
        xb, yb = line(traj, "brier", wb_base)
        axB.plot(xb, yb, color=c, marker="o", ms=3.2, label=label)
        xe, ye = line(traj, "ece", wb_base)
        axE.plot(xe, ye, color=c, marker="o", ms=3.2, label=label)
    axB.axhline(base_b, color=fs.NULL, lw=0.7, ls=":")
    axE.axhline(base_e, color=fs.NULL, lw=0.7, ls=":")
    axB.axhline(0.1451, color=fs.NULL, lw=0.8, ls="--")

    def reflabel(ax, x, y, t, va):
        fs.halo(ax.text(x, y, t, ha="right" if x > 100 else "left", va=va,
                        fontsize=6.8, color=fs.INK), lw=2.6)

    reflabel(axB, 248, base_b, "base (CoT)", "bottom")          # right edge: clear (v9 ends at 150)
    reflabel(axB, 6, 0.1451, "market", "top")                   # lower-left: clear (lines start high)
    reflabel(axE, 248, base_e, "base (CoT)", "bottom")
    axB.set_xlabel("training step"); axB.set_ylabel("Brier (held-out, $n=128$)")
    axE.set_xlabel("training step"); axE.set_ylabel("ECE (held-out, $n=128$)")
    fs.panel_label(axB, "(a)"); fs.panel_label(axE, "(b)")

    handles, labels = axB.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.06), fontsize=7.5, handlelength=1.6, columnspacing=1.8)
    fig.tight_layout(w_pad=2.0, rect=[0, 0, 1, 0.95])
    fs.save(fig, "F2_trajectory")


def compute_ceiling(models, y):
    sys.path.insert(0, str(ROOT / "eval"))
    import teacher_ceiling as TC  # noqa: E402
    d = TC.load("test")
    if not np.array_equal(d["y"], y):
        raise SystemExit("ceiling test outcomes not aligned with preds order")
    rows = [("coarse_phat", "Coarse $\\hat{p}$ (our reward)", d["coarse_phat"]),
            ("nflverse", "nflverse WP model", d["nflverse_wp"])]
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        tr = TC.load("train")
        gbm = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_depth=4,
                                             l2_regularization=1.0).fit(tr["X"], tr["y"].astype(int))
        rows.append(("gbm", "GBM, all features", gbm.predict_proba(d["X"])[:, 1]))
    except Exception as e:  # sklearn absent -> skip, note it
        print(f"(ceiling GBM row skipped: {e})")
    rows.append(("ws1", "Direct RLVR (ours)", models["ws1"]))
    rows.append(("vegas", "Betting market", d["vegas"]))
    out = {}
    for key, name, p in rows:
        p = np.clip(np.asarray(p, float), 0.0, 1.0)
        out[key] = dict(name=name, brier=M.brier_score(p, y),
                        ece=M.expected_calibration_error(p, y, N_BINS))
    return out


def table_ceiling(ceiling):
    lines = [
        "% static-feature Brier ceiling. Only the coarse buckets and the LLM reach the LLM tier;",
        "% richer-feature static models are worse. Vegas's edge is live information.",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Static-feature forecaster & Brier & ECE \\",
        r"\midrule",
    ]
    for key in ["coarse_phat", "nflverse", "gbm", "ws1", "vegas"]:
        if key not in ceiling:
            continue
        c = ceiling[key]
        lines.append(rf"{c['name']} & {c['brier']:.4f} & {c['ece']:.4f} \\")
        if key == "gbm":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "T4_ceiling.tex").write_text("\n".join(lines) + "\n")


# T2 reward-target ablation: IN-TRAINING held-out (n=128), NOT recomputable from the test preds.
# Transcribed from notes/experiments.md (WS1 direct-mode sweeps). Direct prediction, beta=0.01,
# scale_rewards on. p-hat and blend at lr 1e-5; realized-outcome at lr 5e-6 (its most stable; at
# lr 1e-5 its ECE drift is worse). Best-checkpoint Brier / ECE.
LEDGER_T2 = [
    (r"Realized outcome $y\in\{0,1\}$",            "0.166", "0.10"),
    (r"Blend $\tfrac{1}{2}y + \tfrac{1}{2}\hat{p}$", "0.181", "0.121"),
    (r"Empirical rate $\hat{p}$ (ours)",            "0.154", "0.050"),
]


def table_reward():
    lines = [
        "% IN-TRAINING held-out (n=128), direct prediction, beta=0.01, scale_rewards on.",
        "% NOT recomputable from test preds; transcribed from notes/experiments.md.",
        "% p-hat & blend at lr 1e-5; outcome at lr 5e-6 (its most stable run).",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Reward target & Brier & ECE \\",
        r"\midrule",
    ]
    for name, b, e in LEDGER_T2:
        lines.append(rf"{name} & {b} & {e} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "T2_reward.tex").write_text("\n".join(lines) + "\n")


def dump_examples(cap=12):
    """Auditable list of plays where base reasoning is inconsistent but masked-CoT is consistent,
    with the judge's one-line reason. Source for the E1 example box (curated by hand)."""
    recs = [json.loads(l) for l in open(RES / "reasoning_judged_eval.jsonl")]
    def fv(m):
        return ((m.get("verdict") or {}).get("faithfulness") or {}) if isinstance(m, dict) else {}
    out = []
    for r in recs:
        M = r.get("models", {})
        if "base" not in M or "ws2" not in M:
            continue
        if fv(M["base"]).get("v") == "inconsistent" and fv(M["ws2"]).get("v") == "consistent":
            g = r["ground_truth"]
            out.append(
                f"{g['game_id']} | {g['period']} {g['clock']} | {g['posteam']} {g['score']} | "
                f"down {g['down']}&{g['ydstogo']} | {g['field_position']} | "
                f"spread {g['spread']['team']} {g['spread']['points']} | "
                f"phat={g['target_prob']:.2f} outcome={g['actual_outcome']}\n"
                f"  BASE p={M['base']['parsed_prob']}: {fv(M['base']).get('q')}\n"
                f"  WS2  p={M['ws2']['parsed_prob']}: {fv(M['ws2']).get('q')}\n")
    EXAMPLES = PAPER / "examples"
    EXAMPLES.mkdir(exist_ok=True)
    (EXAMPLES / "E1_candidates.txt").write_text(
        f"{len(out)} plays: base inconsistent, masked-CoT consistent\n\n" + "\n".join(out[:cap]))
    return len(out)


def table_decoupling(results):
    pb = results["paired_brier"]
    def fmt(name):
        d = pb[name]
        star = r"$^{\ast}$" if d["sig"] else ""
        sign = "+" if d["delta"] >= 0 else ""
        return rf"{sign}{d['delta']:.4f}{star} & {_ci(d['lo'], d['hi'])}"
    lines = [
        "% paired bootstrap, per-play Brier difference (negative => row-A better). $\\ast$=95% CI excludes 0.",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Paired comparison (Brier) & $\Delta$ & 95\% CI \\",
        r"\midrule",
        rf"Direct $-$ Masked-CoT & {fmt('ws1_vs_ws2')} \\",
        rf"Direct $-$ Market & {fmt('ws1_vs_vegas')} \\",
        rf"Masked-CoT $-$ Market & {fmt('ws2_vs_vegas')} \\",
        rf"Direct $-$ DeepSeek-V4 & {fmt('ws1_vs_deepseek')} \\",
        rf"DeepSeek-V4 $-$ Market & {fmt('deepseek_vs_vegas')} \\",
        r"\bottomrule",
        r"\end{tabular}",
    ]
    (TABLES / "T3_decoupling.tex").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- main
def main():
    import os
    os.chdir(ROOT)                                     # teacher_ceiling uses cwd-relative data paths
    fs.use_style()
    models, y = load_models()
    results = compute(models, y)
    results["faithfulness"] = compute_faithfulness()
    results["ceiling"] = compute_ceiling(models, y)
    json.dump(results, open(PAPER / "results.json", "w"), indent=2)

    fig_reliability(models, y, results)
    fig_triangulation(results)
    fig_trajectory()
    fig_faithfulness(results["faithfulness"])
    table_main(results)
    table_decoupling(results)
    table_ceiling(results["ceiling"])
    table_reward()
    n_ex = dump_examples()
    print(f"E1 candidate examples (base inconsistent, ws2 consistent): {n_ex}")

    # console sanity tie-out vs the ledger
    print(f"n={results['n']} test plays, {N_BINS} bins")
    for k in ["base_cot", "base_direct", "ws1", "ws2", "deepseek", "teacher", "vegas"]:
        r = results["models"][k]
        print(f"  {k:12s} Brier {r['brier']:.4f} [{r['brier_ci'][0]:.4f},{r['brier_ci'][1]:.4f}]"
              f"  ECE {r['ece']:.4f}  MCE {r['mce']:.4f}  acc {r['acc']:.3f}")
    print("paired Brier diffs:")
    for name, d in results["paired_brier"].items():
        print(f"  {name:20s} {d['delta']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}] sig={d['sig']}")
    print("faithfulness (inconsistent rate):")
    for lab in FAITH_CELLS:
        d = results["faithfulness"][lab]
        print(f"  {lab:10s} {d['rate']*100:5.1f}% [{d['lo']*100:.1f},{d['hi']*100:.1f}] n={d['n']}")
    print("static-feature ceiling (test Brier / ECE):")
    for key, c in results["ceiling"].items():
        print(f"  {key:12s} {c['brier']:.4f} / {c['ece']:.4f}")


if __name__ == "__main__":
    main()
