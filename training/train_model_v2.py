"""Train candidate v2 timeout models (offline step of v2).

What changes from v1 (training/train_model.py), and why
--------------------------------------------------------
v1 lesson (docs/RESULTS_DISCUSSION.md)       v2 change
--------------------------------------      -----------------------------------------
Errors are asymmetric: a too-short timeout   Quantile (pinball) loss instead of squared
costs far more than a too-long one.          error. With quantile level q, an under-
                                             prediction costs q/(1-q) times more than an
                                             over-prediction (q = 0.9 -> 9x). The model
                                             also becomes automatically more cautious
                                             where the future is less predictable.
Trained only on traces recorded under the    Training data = round 1 (RFC in control) +
RFC estimator (covariate shift).             round 2 (v1 ML model in control), so the
                                             model sees the situations it creates itself.
Offline accuracy did not predict online      Final choice of quantile level and margin is
behaviour.                                   made end to end on validation transfers
                                             (training/calibrate_v2.py), not offline.

Model: GradientBoostingRegressor (300 trees of depth 4) predicting
log(P95 of the RTTs sent in the next 0.5 s) -- the same label as v1.
Exported to JSON as a "boosting" ensemble, evaluated in pure Python at runtime.

Usage:
    python3 -m training.train_model_v2          # -> models/v2_candidates/q*.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

from experiments.harness import ROOT
from timeout.features import FEATURE_NAMES
from timeout.ml_estimator import MODEL_FORMAT, MODEL_FORMAT_VERSION, ForestModel
from training.dataset import (
    BAND, HORIZON, QUANTILE, RTO_MAX, RTO_MIN, apply_policy, load_runs, proposals_from_prediction,
    stack, summarize,
)

QUANTILE_LEVELS = [0.75, 0.9]
INDEX_FILES = ["runs.csv", "runs_mlctl.csv"]
PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
              min_samples_leaf=40, random_state=0)


def export_boosting(model, path, init, metadata):
    trees = []
    for est in model.estimators_[:, 0]:
        t = est.tree_
        trees.append({"left": t.children_left.tolist(), "right": t.children_right.tolist(),
                      "feature": t.feature.tolist(), "threshold": t.threshold.tolist(),
                      "value": t.value[:, 0, 0].tolist()})
    doc = {"format": MODEL_FORMAT, "format_version": MODEL_FORMAT_VERSION,
           "feature_names": FEATURE_NAMES, "target": "direct", "margin": 1.0,
           "ensemble": "boosting", "init": init, "learning_rate": model.learning_rate,
           "trees": trees, "metadata": metadata}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train v2 candidate models")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out-dir", default=str(ROOT / "models" / "v2_candidates"))
    ap.add_argument("--subsample", type=int, default=3)
    args = ap.parse_args(argv)

    t0 = time.time()
    train_runs = load_runs(args.data_dir, ["train"], index_files=INDEX_FILES)
    val_runs = load_runs(args.data_dir, ["val"], index_files=INDEX_FILES)
    train = stack(train_runs).iloc[::args.subsample].reset_index(drop=True)
    val = stack(val_runs)
    futures = {r.run: r.future_rtts for r in val_runs}
    rounds = {"rfc_control": sum(1 for r in train_runs if not r.run.startswith("mlctl_")),
              "ml_control": sum(1 for r in train_runs if r.run.startswith("mlctl_"))}
    X, y = train[FEATURE_NAMES].to_numpy(), np.log(train["target"].to_numpy())
    Xv = val[FEATURE_NAMES].to_numpy()
    print(f"train rows {len(train)} from {len(train_runs)} runs {rounds}; val rows {len(val)}")

    conv = summarize(val, val["rto_conv"].to_numpy(), futures)
    print(f"  RFC 6298 on val: MAE {conv['mae_ms']:.1f} ms, early-timeout share {conv['spurious_fraction']:.4f}")
    for q in QUANTILE_LEVELS:
        model = GradientBoostingRegressor(loss="quantile", alpha=q, **PARAMS).fit(X, y)
        init = float(model.init_.predict(X[:1]).ravel()[0])
        proposal = proposals_from_prediction(model.predict(Xv), val, "direct")
        m = summarize(val, apply_policy(proposal, val), futures)
        print(f"  q={q:.2f}: MAE {m['mae_ms']:.1f} ms, early-timeout share {m['spurious_fraction']:.4f}, "
              f"under-target {m['under_target_rate']:.3f}")
        imp = dict(zip(FEATURE_NAMES, model.feature_importances_.round(4).tolist()))
        path = os.path.join(args.out_dir, f"q{int(q * 100)}.json")
        export_boosting(model, path, init, {
            "version": 2, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sklearn_version": sklearn.__version__, "model": "GradientBoostingRegressor",
            "loss": "quantile", "quantile_level": q, "params": PARAMS,
            "label": f"log(P{QUANTILE} of RTTs sent in the next {HORIZON} s)",
            "band": list(BAND), "rto_min": RTO_MIN, "rto_max": RTO_MAX,
            "training_rounds": rounds, "train_rows": int(len(train)),
            "train_runs": [r.run for r in train_runs], "val_offline": m, "val_offline_rfc": conv,
            "feature_importances": imp})
        runtime = ForestModel.load(path)
        idx = np.linspace(0, len(Xv) - 1, 1500).astype(int)
        diff = np.max(np.abs(model.predict(Xv[idx]) -
                             np.array([runtime.predict(list(map(float, Xv[i]))) for i in idx])))
        assert diff < 1e-9, f"runtime model differs from scikit-learn by {diff}"
        print(f"    exported {path} ({os.path.getsize(path) / 1e6:.2f} MB), runtime parity {diff:.1e}")
    print(f"done in {time.time() - t0:.0f}s; next: python3 -m training.calibrate_v2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
