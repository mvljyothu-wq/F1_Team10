"""Train the ML timeout estimator (offline) and export it to JSON.

Steps
 1. Load train and validation runs (disjoint seeds, see generate_dataset.py).
 2. For each target formulation and each of two forest sizes, fit a
    RandomForestRegressor on the training runs:
       correction: y = log(target / RTO_conv)   -> RTO = RTO_conv * exp(pred)
       direct    : y = log(target)              -> RTO = exp(pred)
 3. On validation runs only, choose the safety margin: the smallest margin in
    MARGINS whose spurious-timeout fraction (after the runtime safety policy)
    is at most max(conventional's fraction, 1%).
 4. Pick the candidate with the lowest validation MAE at its margin.
 5. Export trees to JSON (no pickle), and check that the pure-Python runtime
    model reproduces scikit-learn's predictions.

The test split is not touched here; see evaluate_model.py.

Usage:
    python3 -m training.train_model
    python3 -m training.train_model --exclude-profile changing --out models/rto_model_lopo_changing.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor

from experiments.harness import ROOT
from timeout.features import FEATURE_NAMES
from timeout.ml_estimator import MODEL_FORMAT, MODEL_FORMAT_VERSION, ForestModel, apply_safety
from training.dataset import (
    BAND, HORIZON, QUANTILE, RTO_MAX, RTO_MIN, apply_policy, load_runs, proposals_from_prediction,
    recent_max_rtt, spurious_fraction_per_row, stack, summarize,
)

MARGINS = [round(1.0 + 0.05 * i, 2) for i in range(21)]        # 1.00 .. 2.00
SPURIOUS_FLOOR = 0.01
CONFIGS = {
    "shallow": dict(max_depth=8, min_samples_leaf=50),
    "deep": dict(max_depth=14, min_samples_leaf=30),
}


def make_target(rows, variant):
    if variant == "correction":
        return np.log(rows["target"].to_numpy() / rows["rto_conv"].to_numpy())
    return np.log(rows["target"].to_numpy())


def calibrate_margin(proposal, rows, futures, limit):
    """Smallest margin whose spurious fraction (after the safety policy) <= limit."""
    for m in MARGINS:
        rto = apply_policy(proposal * m, rows)
        frac = float(np.mean(spurious_fraction_per_row(rows, rto, futures)))
        if frac <= limit:
            return m, frac
    return MARGINS[-1], frac


def export_forest(model, path, target, margin, metadata):
    trees = []
    for est in model.estimators_:
        t = est.tree_
        trees.append({
            "left": t.children_left.tolist(),
            "right": t.children_right.tolist(),
            "feature": t.feature.tolist(),
            "threshold": t.threshold.tolist(),
            "value": t.value[:, 0, 0].tolist(),
        })
    doc = {"format": MODEL_FORMAT, "format_version": MODEL_FORMAT_VERSION,
           "feature_names": FEATURE_NAMES, "target": target, "margin": margin,
           "trees": trees, "metadata": metadata}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train and export the ML RTO model")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "models" / "rto_model.json"))
    ap.add_argument("--exclude-profile", action="append", default=[],
                    help="leave a profile out of training (generalisation test)")
    ap.add_argument("--n-trees", type=int, default=40)
    ap.add_argument("--subsample", type=int, default=3,
                    help="use every k-th training row (adjacent rows are nearly identical)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    t0 = time.time()
    train_runs = load_runs(args.data_dir, ["train"], exclude_profiles=args.exclude_profile)
    val_runs = load_runs(args.data_dir, ["val"], exclude_profiles=args.exclude_profile)
    train = stack(train_runs).iloc[::args.subsample].reset_index(drop=True)
    val = stack(val_runs)
    val_futures = {r.run: r.future_rtts for r in val_runs}
    X_tr = train[FEATURE_NAMES].to_numpy()
    X_val = val[FEATURE_NAMES].to_numpy()
    print(f"train rows {len(train)} from {len(train_runs)} runs; "
          f"val rows {len(val)} from {len(val_runs)} runs; excluded {args.exclude_profile or '-'}")

    conv = summarize(val, val["rto_conv"].to_numpy(), val_futures)
    limit = max(conv["spurious_fraction"], SPURIOUS_FLOOR)
    print(f"conventional on val: MAE {conv['mae_ms']:.1f} ms, spurious {conv['spurious_fraction']:.4f}"
          f" -> spurious limit for ML {limit:.4f}")

    candidates = []
    for variant in ("correction", "direct"):
        y_tr = make_target(train, variant)
        for cname, cfg in CONFIGS.items():
            model = RandomForestRegressor(n_estimators=args.n_trees, max_features=0.6, n_jobs=-1,
                                          random_state=args.seed, **cfg).fit(X_tr, y_tr)
            proposal = proposals_from_prediction(model.predict(X_val), val, variant)
            margin, frac = calibrate_margin(proposal, val, val_futures, limit)
            m = summarize(val, apply_policy(proposal * margin, val), val_futures)
            n_nodes = sum(e.tree_.node_count for e in model.estimators_)
            candidates.append({"variant": variant, "config": cname, "margin": margin,
                               "val": m, "nodes": n_nodes, "model": model})
            print(f"  {variant:10s} {cname:7s} margin {margin:.2f}  MAE {m['mae_ms']:6.1f} ms  "
                  f"spurious {m['spurious_fraction']:.4f}  under-target {m['under_target_rate']:.3f}  "
                  f"nodes {n_nodes}")

    best = min(candidates, key=lambda c: c["val"]["mae_ms"])
    print(f"selected: {best['variant']} / {best['config']} (margin {best['margin']})")

    # Sanity check: vectorised policy == the scalar runtime policy.
    prop = proposals_from_prediction(best["model"].predict(X_val[:500]), val.iloc[:500], best["variant"])
    vec = apply_policy(prop, val.iloc[:500])
    rm = recent_max_rtt(val.iloc[:500])
    conv_vals = val["rto_conv"].to_numpy()[:500]
    scal = [apply_safety(p, c, r, RTO_MIN, RTO_MAX, *BAND) for p, c, r in zip(prop, conv_vals, rm)]
    assert np.allclose(vec, scal), "vectorised policy differs from runtime policy"

    importances = dict(zip(FEATURE_NAMES, best["model"].feature_importances_.round(4).tolist()))
    metadata = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sklearn_version": sklearn.__version__,
        "variant": best["variant"], "config": best["config"], "params": CONFIGS[best["config"]],
        "n_trees": args.n_trees, "max_features": 0.6,
        "horizon_s": HORIZON, "quantile": QUANTILE, "band": list(BAND),
        "rto_min": RTO_MIN, "rto_max": RTO_MAX, "spurious_limit": limit,
        "train_runs": [r.run for r in train_runs], "val_runs": [r.run for r in val_runs],
        "excluded_profiles": args.exclude_profile,
        "train_rows": int(len(train)), "val_rows": int(len(val)),
        "val_conventional": conv,
        "candidates": [{k: v for k, v in c.items() if k != "model"} for c in candidates],
        "feature_importances": importances,
    }
    export_forest(best["model"], args.out, best["variant"], best["margin"], metadata)

    # Parity: the JSON model evaluated in pure Python must match scikit-learn.
    runtime = ForestModel.load(args.out)
    idx = np.linspace(0, len(X_val) - 1, 2000).astype(int)
    sk = best["model"].predict(X_val[idx])
    py = np.array([runtime.predict(list(map(float, X_val[i]))) for i in idx])
    max_diff = float(np.max(np.abs(sk - py)))
    assert max_diff < 1e-9, f"runtime model differs from scikit-learn by {max_diff}"
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"exported {args.out} ({size_mb:.1f} MB); runtime/sklearn max diff {max_diff:.2e}; "
          f"took {time.time() - t0:.0f}s")
    print("feature importances:", ", ".join(f"{k}={v:.3f}" for k, v in
                                           sorted(importances.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
