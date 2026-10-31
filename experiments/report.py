"""Turn experiment outputs into plots and a results table (results/RESULTS.md).

Inputs : results/trials.csv (run_experiments.py), results/traces/*.csv,
         results/offline_eval.csv + .json (training/evaluate_model.py), the model JSON.
Outputs: results/plots/*.png, results/summary.csv, results/RESULTS.md

Every number in the output is computed from those files; nothing is typed in.

Usage:
    python3 -m experiments.report
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from experiments import plotstyle
from experiments.harness import ROOT
from experiments.metrics import paired

# (column, label, unit scale, higher_is_better)
METRICS = [
    ("goodput_MBps", "Goodput (MB/s)", 1, True),
    ("duration", "Transfer time (s)", 1, False),
    ("retransmissions", "Retransmissions", 1, False),
    ("unnecessary_retx", "Unnecessary retransmissions", 1, False),
    ("recovery_ms", "Loss-recovery delay (ms)", 1, False),
    ("timeouts", "Timeout events", 1, False),
    ("overhead_pct", "Traffic overhead (%)", 1, False),
    ("cpu_seconds", "Sender CPU (s)", 1, False),
]


def load_trials(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["goodput_MBps"] = df["goodput_Bps"] / 1e6
    df["recovery_ms"] = df["loss_recovery_mean"] * 1000
    # bytes beyond the file itself (headers, retransmissions, control packets) as % of file size
    file_bytes = df["goodput_Bps"] * df["duration"]
    df["overhead_pct"] = 100 * (df["bytes_sent"] - file_bytes) / file_bytes.where(file_bytes > 0)
    df["loss_rate"] = df["data_copies_lost"] / df["data_packets_sent"]
    return df


COMPARISONS = [("rto", "ml"), ("rto", "rto_k2"), ("rto_k2", "ml"), ("rto", "ml_v2"), ("ml", "ml_v2"),
               ("rto_k2", "ml_v2"), ("rto", "rto_rack"), ("rto", "ml_v2_rack"), ("rto_rack", "ml_v2_rack")]


def summarize(df: pd.DataFrame, base: str = "rto", cand: str = "ml") -> pd.DataFrame:
    """Paired statistics of `cand` relative to `base` (columns conv_* = base, ml_* = cand)."""
    rows = []
    if base not in set(df["mode"]) or cand not in set(df["mode"]):
        return pd.DataFrame(rows)
    for profile in df["profile"].unique():
        sub = df[df["profile"] == profile]
        wide = sub.pivot(index="seed", columns="mode")
        for col, label, _, better in METRICS:
            if col not in df:
                continue
            conv, ml = wide[col][base], wide[col][cand]
            ok = conv.notna() & ml.notna()
            if ok.sum() == 0:
                continue
            rows.append({"comparison": f"{cand}_vs_{base}", "profile": profile,
                         "metric": col, "label": label,
                         "higher_is_better": better, **paired(conv[ok], ml[ok])})
    return pd.DataFrame(rows)


def bar_compare(df, col, title, path, ylabel=None):
    plt = plotstyle.plt
    profs = list(df["profile"].unique())
    g = df.groupby(["profile", "mode"])[col]
    means, stds = g.mean(), g.std()
    x = np.arange(len(profs))
    modes = [m for m in plotstyle.MODE_ORDER if m in set(df["mode"])]
    w = 0.84 / len(modes)
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    for i, mode in enumerate(modes):
        m = [means.get((p, mode), np.nan) for p in profs]
        s = [stds.get((p, mode), np.nan) for p in profs]
        ax.bar(x + (i - (len(modes) - 1) / 2) * w, m, w * 0.94, yerr=s, capsize=3, color=plotstyle.COLORS[mode],
               label=plotstyle.LABELS[mode], error_kw={"elinewidth": 1, "ecolor": plotstyle.MUTED})
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("_", "\n") for p in profs])
    ax.set_title(title, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend(loc="lower right", ncol=len(modes), bbox_to_anchor=(1.0, 1.0), fontsize=8)
    ax.text(0.0, -0.22, "Bars: mean over paired seeds; whiskers: one standard deviation.",
            transform=ax.transAxes, fontsize=8, color=plotstyle.MUTED)
    plotstyle.save(fig, path)


def timeline(traces_dir, profile, seed, path):
    plt = plotstyle.plt
    files = {m: os.path.join(traces_dir, f"{profile}_s{seed}_{m}.csv") for m in ("rto", "ml")}
    if not all(os.path.exists(f) for f in files.values()):
        return False
    data = {m: pd.read_csv(f) for m, f in files.items()}
    # one shared y-range so the two panels are directly comparable
    ymax = max(1.0, max(np.nanpercentile(d["rto_final"] * 1000, 99) for d in data.values()) * 1.25)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, sharey=True)
    for ax, mode in zip(axes, ("rto", "ml")):
        d = data[mode]
        s = d[d["event"] == "sample"]
        to = d[d["event"] == "timeout"]
        ax.scatter(s["t"], s["rtt"] * 1000, s=5, color=plotstyle.MUTED, alpha=0.4,
                   edgecolors="none", label="RTT samples")
        # a timeout keeps its value until the next update: draw it as steps
        ax.step(d["t"], d["rto_final"] * 1000, where="post", color=plotstyle.COLORS[mode],
                linewidth=1.5, label="Selected timeout (incl. backoff)")
        if mode == "ml":
            ax.step(s["t"], s["rto_conv"] * 1000, where="post", color=plotstyle.COLORS["rto"],
                    linewidth=1, linestyle="--", label="Conventional RTO (for reference)")
        ax.scatter(to["t"], np.full(len(to), ymax * 0.96), marker="|", s=40,
                   color=plotstyle.TEXT, label=f"Timeout events ({len(to)})")
        ax.set_ylim(0, ymax)
        ax.set_ylabel("ms")
        ax.set_title(f"{plotstyle.LABELS[mode]} - profile '{profile}', seed {seed}", loc="left")
        ax.legend(loc="upper left", bbox_to_anchor=(0, 0.92), ncol=2, fontsize=8)
    axes[1].set_xlabel("Time since transfer start (s)")
    plotstyle.save(fig, path)
    return True


def heatmap(summary, path, title="ML vs conventional: change in the mean (blue = better for ML, red = worse)"):
    plt = plotstyle.plt
    metrics = [m for m in ["goodput_MBps", "duration", "retransmissions", "unnecessary_retx",
                           "recovery_ms", "overhead_pct"] if m in set(summary["metric"])]
    profs = list(summary["profile"].unique())
    labels = {m[0]: m[1] for m in METRICS}
    pct = np.full((len(profs), len(metrics)), np.nan)
    good = np.full_like(pct, np.nan)        # positive = better for ML
    sig = np.zeros_like(pct, dtype=bool)
    for i, p in enumerate(profs):
        for j, m in enumerate(metrics):
            r = summary[(summary.profile == p) & (summary.metric == m)]
            if len(r):
                r = r.iloc[0]
                pct[i, j] = r["pct_change"]
                good[i, j] = r["pct_change"] * (1 if r["higher_is_better"] else -1)
                sig[i, j] = r["p_value"] is not None and r["p_value"] < 0.05
    cmap = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])
    lim = 25.0          # colour saturates at +-25% so moderate effects stay visible
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(profs) + 1.9))
    ax.imshow(np.clip(good, -lim, lim), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(len(profs)):
        for j in range(len(metrics)):
            if np.isfinite(pct[i, j]):
                ax.text(j, i, f"{pct[i, j]:+.0f}%" + ("*" if sig[i, j] else ""), ha="center",
                        va="center", fontsize=9, color=plotstyle.TEXT)
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([labels[m].replace(" (", "\n(") for m in metrics], fontsize=8)
    ax.set_yticks(range(len(profs)))
    ax.set_yticklabels(profs)
    ax.grid(False)
    ax.set_title(title, loc="left", fontsize=11)
    ax.text(0, -0.8 / (len(profs) + 1), "* paired Wilcoxon p < 0.05", transform=ax.transAxes,
            fontsize=8, color=plotstyle.MUTED, va="top")
    plotstyle.save(fig, path)


def tradeoff(df, path):
    """Goodput vs unnecessary retransmissions per profile: is a strategy better,
    or just further along the same aggressiveness trade-off?"""
    plt = plotstyle.plt
    profs = [p for p in df["profile"].unique() if p != "stable"]
    modes = [m for m in plotstyle.MODE_ORDER if m in set(df["mode"])]
    cols = 3
    rows = int(np.ceil(len(profs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.3 * rows))
    axes = np.atleast_1d(axes).ravel()
    g = df.groupby(["profile", "mode"])
    for ax, p in zip(axes, profs):
        for mode in modes:
            d = g.get_group((p, mode))
            ax.errorbar(d["unnecessary_retx"].mean(), d["goodput_MBps"].mean(),
                        xerr=d["unnecessary_retx"].std(), yerr=d["goodput_MBps"].std(),
                        fmt="o", markersize=8, color=plotstyle.COLORS[mode], ecolor=plotstyle.COLORS[mode],
                        elinewidth=1, capsize=2, label=plotstyle.LABELS[mode],
                        markeredgecolor=plotstyle.SURFACE, markeredgewidth=2)
        ax.set_title(p, loc="left", fontsize=10)
        ax.grid(True, axis="both")
        ax.set_xlabel("unnecessary retransmissions", fontsize=8)
        ax.set_ylabel("goodput MB/s", fontsize=8)
    for ax in axes[len(profs):]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(modes), bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Up-left is better: more goodput for less wasted retransmission (mean +- 1 sd)",
                 y=0.94, fontsize=10, color=plotstyle.MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fmt(v, digits=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:.{digits}f}"


def _k2_table(lines, title, summ, base_name, cand_name):
    lines += [f"## {title}", "",
              f"| Profile | Goodput {base_name} | {cand_name} | change | p | "
              f"Unneeded retx {base_name} | {cand_name} | change | p |",
              "|---|---|---|---|---|---|---|---|---|"]
    for p in summ["profile"].unique():
        gp = summ[(summ.profile == p) & (summ.metric == "goodput_MBps")].iloc[0]
        un = summ[(summ.profile == p) & (summ.metric == "unnecessary_retx")].iloc[0]
        unc = f"{un['pct_change']:+.0f}%" if np.isfinite(un["pct_change"]) else "-"
        lines.append(f"| {p} | {gp['conv_mean']:.3f} | {gp['ml_mean']:.3f} | {gp['pct_change']:+.1f}% | "
                     f"{fmt(gp['p_value'], 3)} | {un['conv_mean']:.1f} | {un['ml_mean']:.1f} | {unc} | "
                     f"{fmt(un['p_value'], 3)} |")
    lines.append("")


def write_markdown(df, summary, offline, offline_meta, model_meta, plots, path, vs_k2=None, k2_vs_rto=None):
    lines = ["# Results", "",
             "Generated by `python3 -m experiments.report` from `results/trials.csv` and "
             "`results/offline_eval.*`. Do not edit by hand; rerun the pipeline instead.", ""]
    n_seeds = df.groupby("profile")["seed"].nunique()
    ok = df["success"].sum()
    ver = df["verified"].sum()
    lines += ["## Setup", "",
              f"- {df.groupby(['profile', 'seed']).ngroups} (profile, seed) combinations x "
              f"{df['mode'].nunique()} timeout modes ({', '.join(sorted(df['mode'].unique()))}) = {len(df)} transfers; "
              f"{n_seeds.min()} seeds per profile; file size {int(df['goodput_Bps'].mul(df['duration']).median()):,} bytes.",
              f"- Successful transfers: {ok}/{len(df)}; independently SHA-256-verified: {ver}/{len(df)}.",
              f"- ML model: target `{model_meta.get('variant')}`, config `{model_meta.get('config')}`, "
              f"margin {offline_meta.get('margin')}, trained on {len(model_meta.get('train_runs', []))} runs.",
              "- All runs on one machine over loopback through the user-space impairment proxy "
              "(see limitations).", ""]

    lines += ["## End-to-end comparison (mean per profile)", "",
              "| Profile | Goodput MB/s conv | ML | change [95% CI] | p | Retx conv | ML | "
              "Unneeded retx conv | ML | Recovery ms conv | ML |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in summary["profile"].unique():
        def g(metric):
            r = summary[(summary.profile == p) & (summary.metric == metric)]
            return r.iloc[0] if len(r) else None
        gp, rt, un, rc = g("goodput_MBps"), g("retransmissions"), g("unnecessary_retx"), g("recovery_ms")
        ci = (f"{100 * gp['diff_ci_lo'] / gp['conv_mean']:+.1f}% .. {100 * gp['diff_ci_hi'] / gp['conv_mean']:+.1f}%"
              if gp is not None else "-")
        lines.append(
            f"| {p} | {fmt(gp['conv_mean'], 3)} | {fmt(gp['ml_mean'], 3)} | {gp['pct_change']:+.1f}% "
            f"[{ci}] | {fmt(gp['p_value'], 3)} | {fmt(rt['conv_mean'], 0)} | {fmt(rt['ml_mean'], 0)} | "
            f"{fmt(un['conv_mean'], 0)} | {fmt(un['ml_mean'], 0)} | "
            f"{fmt(rc['conv_mean'], 0) if rc is not None else '-'} | {fmt(rc['ml_mean'], 0) if rc is not None else '-'} |")
    lines += ["", "Change = (ML - conventional) / conventional, paired by seed. "
              "p = exact two-sided Wilcoxon signed-rank test on the paired differences. "
              "With 10 pairs the smallest attainable p is about 0.002; treat p near 0.05 as weak evidence.", ""]
    arms = [m for m in plotstyle.MODE_ORDER if m in set(df["mode"])]
    if len(arms) > 3:
        g = df.groupby(["profile", "mode"])
        lines += ["## All strategies (mean per profile)", "",
                  "Goodput in MB/s / unnecessary retransmissions per 1 MiB transfer.", "",
                  "| Profile | " + " | ".join(plotstyle.LABELS[m] for m in arms) + " |",
                  "|---|" + "---|" * len(arms)]
        for p in df["profile"].unique():
            cells = []
            for m in arms:
                if (p, m) in g.groups:
                    d = g.get_group((p, m))
                    cells.append(f"{d['goodput_MBps'].mean():.3f} / {d['unnecessary_retx'].mean():.0f}")
                else:
                    cells.append("-")
            lines.append(f"| {p} | " + " | ".join(cells) + " |")
        lines.append("")
        allc = pd.concat([summarize(df, a, b) for a, b in COMPARISONS], ignore_index=True)
        lines += ["## Paired comparisons (goodput change, unnecessary-retransmission change)", "",
                  "| Comparison | " + " | ".join(df["profile"].unique()) + " |",
                  "|---|" + "---|" * df["profile"].nunique()]
        for a, b in COMPARISONS:
            sub = allc[allc.comparison == f"{b}_vs_{a}"]
            if not len(sub):
                continue
            cells = []
            for p in df["profile"].unique():
                gp = sub[(sub.profile == p) & (sub.metric == "goodput_MBps")]
                un = sub[(sub.profile == p) & (sub.metric == "unnecessary_retx")]
                if not len(gp):
                    cells.append("-")
                    continue
                gp, un = gp.iloc[0], un.iloc[0]
                star = "*" if gp["p_value"] is not None and gp["p_value"] < 0.05 else ""
                unc = f"{un['pct_change']:+.0f}%" if np.isfinite(un["pct_change"]) else "0"
                cells.append(f"{gp['pct_change']:+.1f}%{star} / {unc}")
            lines.append(f"| {plotstyle.LABELS[b]} vs {plotstyle.LABELS[a]} | " + " | ".join(cells) + " |")
        lines += ["", "* = goodput difference significant (paired Wilcoxon p < 0.05).", ""]
    if vs_k2 is not None and len(vs_k2):
        lines += ["Ablation: is the ML estimator smarter, or just more aggressive? The third arm "
                  "uses the RFC formula with K=2 instead of 4 (a shorter, more aggressive timeout) "
                  "on the same seeds. It was run after the first two arms (not interleaved).", ""]
        _k2_table(lines, "Ablation: K=2 conventional vs standard (K=4)", k2_vs_rto, "K=4", "K=2")
        _k2_table(lines, "Ablation: ML vs K=2 conventional", vs_k2, "K=2", "ML")

    lines += ["## Offline timeout-prediction evaluation (held-out test runs)", "",
              "| Profile | Estimator | MAE ms | Spurious-timeout fraction | RTO < target rate | Mean excess ms |",
              "|---|---|---|---|---|---|"]
    for r in offline.itertuples():
        if r.estimator in ("conventional", "recent_max", "ml"):
            lines.append(f"| {r.profile} | {r.estimator} | {r.mae_ms:.1f} | {r.spurious_fraction:.4f} | "
                         f"{r.under_target_rate:.3f} | {r.mean_excess_ms:.1f} |")
    if "lopo" in offline_meta:
        l = offline_meta["lopo"]
        lines += ["", f"Leave-one-profile-out on `{l['profile']}` (model never saw this profile):", "",
                  "| Estimator | MAE ms | Spurious fraction |", "|---|---|---|"]
        for k in ("conventional", "ml_trained_with_profile", "ml_trained_without_profile"):
            lines.append(f"| {k} | {l[k]['mae_ms']:.1f} | {l[k]['spurious_fraction']:.4f} |")
    lines += ["", "## Plots", ""] + [f"- `plots/{os.path.basename(p)}`" for p in plots] + [""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build plots and RESULTS.md")
    ap.add_argument("--results-dir", default=str(ROOT / "results"))
    ap.add_argument("--model", default=str(ROOT / "models" / "rto_model.json"))
    ap.add_argument("--timeline-profile", default="changing")
    args = ap.parse_args(argv)

    plotstyle.setup()
    rd = args.results_dir
    pd_ = os.path.join(rd, "plots")
    os.makedirs(pd_, exist_ok=True)
    df = load_trials(os.path.join(rd, "trials.csv"))
    all_summaries = pd.concat([summarize(df, a, b) for a, b in COMPARISONS], ignore_index=True)
    all_summaries.to_csv(os.path.join(rd, "summary.csv"), index=False)
    summary = all_summaries[all_summaries.comparison == "ml_vs_rto"]
    vs_k2 = all_summaries[all_summaries.comparison == "ml_vs_rto_k2"]

    plots = []
    for col, title, name, ylabel in [
        ("goodput_MBps", "Goodput (file data delivered per second)", "goodput.png", "MB/s"),
        ("retransmissions", "Retransmissions per 1 MiB transfer", "retransmissions.png", "packets"),
        ("duration", "Transfer duration", "duration.png", "seconds"),
        ("unnecessary_retx", "Unnecessary retransmissions (proxy ground truth)", "unnecessary_retx.png", "packets"),
        ("recovery_ms", "Mean loss-recovery delay (lost copy -> next copy)", "recovery_delay.png", "ms"),
    ]:
        p = os.path.join(pd_, name)
        bar_compare(df, col, title, p, ylabel)
        plots.append(p)
    for prof in [args.timeline_profile, "jitter_heavy", "lossy"]:
        seed = int(df[df.profile == prof]["seed"].min()) if (df.profile == prof).any() else None
        p = os.path.join(pd_, f"timeline_{prof}.png")
        if seed is not None and timeline(os.path.join(rd, "traces"), prof, seed, p):
            plots.append(p)
    p = os.path.join(pd_, "change_heatmap.png")
    heatmap(summary, p)
    plots.append(p)
    if len(vs_k2):
        p = os.path.join(pd_, "change_heatmap_vs_k2.png")
        heatmap(vs_k2, p, "ML vs aggressive conventional (K=2): change in the mean "
                          "(blue = better for ML)")
        plots.append(p)
        p = os.path.join(pd_, "tradeoff.png")
        tradeoff(df, p)
        plots.append(p)
    for extra in ("offline_error_by_profile.png", "offline_timeout_vs_target.png", "feature_importance.png"):
        if os.path.exists(os.path.join(pd_, extra)):
            plots.append(os.path.join(pd_, extra))

    offline = pd.read_csv(os.path.join(rd, "offline_eval.csv"))
    with open(os.path.join(rd, "offline_eval.json")) as f:
        offline_meta = json.load(f)
    with open(args.model) as f:
        model_meta = json.load(f).get("metadata", {})
    write_markdown(df, summary, offline, offline_meta, model_meta, plots, os.path.join(rd, "RESULTS.md"),
                   vs_k2=vs_k2, k2_vs_rto=all_summaries[all_summaries.comparison == "rto_k2_vs_rto"])
    print(all_summaries[all_summaries.metric.isin(["goodput_MBps", "retransmissions", "unnecessary_retx", "recovery_ms"])]
          [["comparison", "profile", "metric", "conv_mean", "ml_mean", "pct_change", "p_value"]].round(4).to_string(index=False))
    print(f"wrote {rd}/RESULTS.md, summary.csv and {len(plots)} plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
