"""Offline evaluation of timeout estimators on the held-out TEST runs.

Compares, on exactly the same decision points:
  conventional : RFC 6298 RTO (as logged by the sender)
  recent_max   : a trivial baseline -- the largest of the last 16 RTTs
  ml_raw       : ML proposal x margin, only clipped to [rto_min, rto_max]
  ml           : ML proposal after the runtime safety policy (what the sender uses)

Predictions go through the pure-Python runtime model (timeout/ml_estimator.py),
so this evaluates the code that actually runs in the sender.

Metrics (per row = per decision point, target = P95 of future RTTs):
  mae_ms            mean |RTO - target|
  spurious_fraction mean fraction of future delivered packets with RTT > RTO
                    (they would time out although their ACK was on the way)
  under_target_rate how often RTO < target
  mean_excess_ms    average amount by which RTO exceeds the target, when it does
                    (extra waiting before recovering a real loss)

A lower MAE does not automatically mean a better transport: errors are
asymmetric (too short -> spurious retransmissions + backoff; too long -> slower
recovery), so end-to-end experiments are the real test.

Usage:
    python3 -m training.evaluate_model
    python3 -m training.evaluate_model --lopo-model models/rto_model_lopo_changing.json --lopo-profile changing
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from experiments import plotstyle
from experiments.harness import ROOT
from timeout.features import FEATURE_NAMES
from timeout.ml_estimator import ForestModel
from training.dataset import (
    RTO_MAX, RTO_MIN, apply_policy, load_runs, proposals_from_prediction, recent_max_rtt, stack,
    summarize,
)


def ml_rtos(model: ForestModel, rows: pd.DataFrame):
    X = rows[FEATURE_NAMES].to_numpy()
    pred = np.array([model.predict(list(map(float, x))) for x in X])
    proposal = proposals_from_prediction(pred, rows, model.target) * model.margin
    return np.clip(proposal, RTO_MIN, RTO_MAX), apply_policy(proposal, rows)


def evaluate(rows, futures, model):
    raw, final = ml_rtos(model, rows)
    estimators = {
        "conventional": rows["rto_conv"].to_numpy(),
        "recent_max": np.clip(recent_max_rtt(rows), RTO_MIN, RTO_MAX),
        "ml_raw": raw,
        "ml": final,
    }
    out = []
    for prof in ["ALL"] + sorted(rows["profile"].unique()):
        mask = np.ones(len(rows), bool) if prof == "ALL" else (rows["profile"] == prof).to_numpy()
        sub = rows[mask].reset_index(drop=True)
        for name, rto in estimators.items():
            out.append({"profile": prof, "estimator": name, **summarize(sub, rto[mask], futures)})
    return pd.DataFrame(out), estimators


def plot_offline(df, path):
    plt = plotstyle.plt
    profs = [p for p in df["profile"].unique() if p != "ALL"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    width = 0.38
    x = np.arange(len(profs))
    for ax, metric, title, scale in ((axes[0], "mae_ms", "Timeout error vs target (MAE, ms)", 1),
                                     (axes[1], "spurious_fraction",
                                      "Packets that would time out spuriously (%)", 100)):
        for i, (est, key) in enumerate((("conventional", "rto"), ("ml", "ml"))):
            vals = [df[(df.profile == p) & (df.estimator == est)][metric].iloc[0] * scale for p in profs]
            ax.bar(x + (i - 0.5) * width, vals, width * 0.94, color=plotstyle.COLORS[key],
                   label=plotstyle.LABELS[key])
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("_", "\n") for p in profs])
        ax.set_title(title, loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, bbox_to_anchor=(1.0, 1.0))
    fig.subplots_adjust(top=0.82)
    plotstyle.save(fig, path)


def plot_scatter(rows, estimators, path, n=4000):
    plt = plotstyle.plt
    rng = np.random.default_rng(0)
    idx = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    y = rows["target"].to_numpy()[idx] * 1000
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    for ax, (est, key) in zip(axes, (("conventional", "rto"), ("ml", "ml"))):
        ax.scatter(y, estimators[est][idx] * 1000, s=8, alpha=0.35, color=plotstyle.COLORS[key],
                   edgecolors="none")
        lim = [max(1, y.min() * 0.8), y.max() * 1.6]
        ax.plot(lim, lim, color=plotstyle.MUTED, linewidth=1, linestyle="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ticks = [10, 20, 50, 100, 200, 500, 1000]
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(FixedLocator(ticks))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(NullFormatter())
        ax.set_title(plotstyle.LABELS[key], loc="left")
        ax.set_xlabel("Target: P95 of next 0.5 s of RTTs (ms)")
        ax.grid(True, which="major", axis="both")
    axes[0].set_ylabel("Chosen timeout (ms)")
    fig.suptitle("Timeout vs target on held-out test runs (points above the dashed line wait longer than needed)",
                 fontsize=10, color=plotstyle.MUTED)
    plotstyle.save(fig, path)


def plot_importance(model: ForestModel, path):
    plt = plotstyle.plt
    imp = model.metadata.get("feature_importances", {})
    if not imp:
        return
    items = sorted(imp.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh([k for k, _ in items], [v for _, v in items], color=plotstyle.COLORS["ml"], height=0.6)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_title("Feature importance (impurity-based)", loc="left")
    plotstyle.save(fig, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Offline evaluation of timeout estimators")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--model", default=str(ROOT / "models" / "rto_model.json"))
    ap.add_argument("--lopo-model", help="model trained without --lopo-profile")
    ap.add_argument("--lopo-profile", default="changing")
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    args = ap.parse_args(argv)

    plotstyle.setup()
    plots = os.path.join(args.out_dir, "plots")
    os.makedirs(plots, exist_ok=True)
    model = ForestModel.load(args.model)
    runs = load_runs(args.data_dir, ["test"])
    rows = stack(runs)
    futures = {r.run: r.future_rtts for r in runs}
    print(f"test rows {len(rows)} from {len(runs)} runs; model target={model.target} margin={model.margin}")

    df, estimators = evaluate(rows, futures, model)
    df.to_csv(os.path.join(args.out_dir, "offline_eval.csv"), index=False)
    pd.set_option("display.width", 160)
    print(df.round(4).to_string(index=False))

    report = {"model": os.path.relpath(args.model, ROOT), "target": model.target,
              "margin": model.margin, "test_rows": int(len(rows)), "test_runs": len(runs)}
    if args.lopo_model:
        lopo = ForestModel.load(args.lopo_model)
        sub_runs = [r for r in runs if r.profile == args.lopo_profile]
        sub = stack(sub_runs)
        _, lopo_final = ml_rtos(lopo, sub)
        _, full_final = ml_rtos(model, sub)
        report["lopo"] = {
            "profile": args.lopo_profile,
            "conventional": summarize(sub, sub["rto_conv"].to_numpy(), futures),
            "ml_trained_with_profile": summarize(sub, full_final, futures),
            "ml_trained_without_profile": summarize(sub, lopo_final, futures),
        }
        print(f"\nLeave-one-profile-out on '{args.lopo_profile}':")
        for k, v in report["lopo"].items():
            if isinstance(v, dict):
                print(f"  {k:28s} MAE {v['mae_ms']:6.1f} ms  spurious {v['spurious_fraction']:.4f}")
    with open(os.path.join(args.out_dir, "offline_eval.json"), "w") as f:
        json.dump(report, f, indent=2)

    plot_offline(df, os.path.join(plots, "offline_error_by_profile.png"))
    plot_scatter(rows, estimators, os.path.join(plots, "offline_timeout_vs_target.png"))
    plot_importance(model, os.path.join(plots, "feature_importance.png"))
    print(f"wrote {args.out_dir}/offline_eval.csv, offline_eval.json and plots/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
