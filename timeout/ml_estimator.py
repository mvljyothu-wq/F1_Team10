"""Runtime side of the ML timeout estimator.

* ForestModel   -- loads a tree ensemble exported to JSON by training/train_model.py
                   (v1: random forest = average of trees; v2: gradient boosting =
                   init + learning_rate x sum of trees) and evaluates it in pure
                   Python (no scikit-learn needed at runtime).
* apply_safety  -- the bounding policy every ML proposal goes through.
* MLTimeoutEstimator -- turns features into a bounded timeout, or reports why it
                   must fall back to the conventional estimator.

Why JSON instead of pickle/joblib: loading a pickle can execute arbitrary
code, so a tampered model file would be a code-execution vulnerability. A JSON
file can only contain numbers and lists, and the loader validates its
structure (including that trees cannot loop) before use.
"""

from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple

from timeout.features import FEATURE_NAMES

MODEL_FORMAT = "rto-forest"
MODEL_FORMAT_VERSION = 1
MAX_MODEL_BYTES = 64 * 1024 * 1024
MAX_TREES = 500
MAX_NODES_PER_TREE = 200_000
PRED_LOG_CLIP = 8.0            # exp(+-8) ~ factor 3000; anything beyond is nonsense


class ModelError(Exception):
    """The model file is missing, malformed or incompatible."""


def _f32(value: float) -> float:
    # scikit-learn trees compare float32 inputs; mimic that for identical results.
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


class ForestModel:
    def __init__(self, trees, feature_names: List[str], target: str, margin: float, metadata: dict,
                 ensemble: str = "forest", init: float = 0.0, learning_rate: float = 1.0):
        self.trees = trees                 # list of (left, right, feature, threshold, value)
        self.ensemble = ensemble           # "forest" (mean of trees) or "boosting" (init + lr * sum)
        self.init = init
        self.learning_rate = learning_rate
        self.feature_names = feature_names
        self.target = target               # "correction" or "direct"
        self.margin = margin
        self.metadata = metadata

    @classmethod
    def load(cls, path: str) -> "ForestModel":
        try:
            size = os.path.getsize(path)
        except OSError as exc:
            raise ModelError(f"cannot open model file: {exc}") from None
        if size > MAX_MODEL_BYTES:
            raise ModelError("model file too large")
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError) as exc:
            raise ModelError(f"cannot parse model file: {exc}") from None
        return cls.from_dict(doc)

    @classmethod
    def from_dict(cls, doc) -> "ForestModel":
        if not isinstance(doc, dict):
            raise ModelError("model root must be an object")
        if doc.get("format") != MODEL_FORMAT or doc.get("format_version") != MODEL_FORMAT_VERSION:
            raise ModelError("unsupported model format/version")
        if doc.get("feature_names") != FEATURE_NAMES:
            raise ModelError("model feature list does not match timeout/features.py")
        target = doc.get("target")
        if target not in ("correction", "direct"):
            raise ModelError("target must be 'correction' or 'direct'")
        margin = doc.get("margin")
        if not isinstance(margin, (int, float)) or not 0.5 <= margin <= 5.0:
            raise ModelError("margin missing or out of range")
        raw_trees = doc.get("trees")
        if not isinstance(raw_trees, list) or not 1 <= len(raw_trees) <= MAX_TREES:
            raise ModelError("trees must be a non-empty list")
        trees = [cls._validate_tree(t, len(FEATURE_NAMES)) for t in raw_trees]
        ensemble = doc.get("ensemble", "forest")
        if ensemble not in ("forest", "boosting"):
            raise ModelError("ensemble must be 'forest' or 'boosting'")
        init, lr = 0.0, 1.0
        if ensemble == "boosting":
            init, lr = doc.get("init"), doc.get("learning_rate")
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (init, lr)) or not 0 < lr <= 1:
                raise ModelError("boosting model needs finite init and 0 < learning_rate <= 1")
        return cls(trees, FEATURE_NAMES, target, float(margin), doc.get("metadata", {}),
                   ensemble, float(init), float(lr))

    @staticmethod
    def _validate_tree(t, n_features: int):
        try:
            left, right, feat, thr, val = (t["left"], t["right"], t["feature"],
                                           t["threshold"], t["value"])
        except (TypeError, KeyError):
            raise ModelError("tree is missing fields") from None
        n = len(left)
        if not 1 <= n <= MAX_NODES_PER_TREE or not all(len(a) == n for a in (right, feat, thr, val)):
            raise ModelError("tree arrays have inconsistent lengths")
        for i in range(n):
            l, r = left[i], right[i]
            if not (isinstance(l, int) and isinstance(r, int)):
                raise ModelError("child indices must be integers")
            if l == -1 and r == -1:                       # leaf
                if not isinstance(val[i], (int, float)) or not math.isfinite(val[i]):
                    raise ModelError("leaf value must be finite")
                continue
            # Children must come after their parent: this guarantees that
            # traversal always terminates (no cycles) and stays in range.
            if not (i < l < n and i < r < n):
                raise ModelError("invalid child index")
            f = feat[i]
            if not isinstance(f, int) or not 0 <= f < n_features:
                raise ModelError("invalid feature index")
            if not isinstance(thr[i], (int, float)) or not math.isfinite(thr[i]):
                raise ModelError("threshold must be finite")
        return (list(left), list(right), list(feat), [float(x) for x in thr], [float(x) for x in val])

    def predict(self, x: List[float]) -> float:
        if len(x) != len(self.feature_names):
            raise ValueError("wrong feature vector length")
        x32 = [_f32(v) for v in x]
        total = 0.0
        for left, right, feat, thr, val in self.trees:
            node = 0
            while left[node] != -1:
                node = left[node] if x32[feat[node]] <= thr[node] else right[node]
            total += val[node]
        if self.ensemble == "boosting":
            return self.init + self.learning_rate * total
        return total / len(self.trees)


def apply_safety(proposal: float, rto_conv: float, recent_max_rtt: float,
                 rto_min: float, rto_max: float, band_lo: float, band_hi: float) -> float:
    """Bound an ML proposal (seconds). See docs/DESIGN.md section 8.

    lower bound: rto_min, band_lo * RTO_conv, and the largest RTT just observed
                 (a timeout below an RTT we have just seen guarantees spurious
                 retransmissions)
    upper bound: rto_max and band_hi * RTO_conv
    """
    lo = max(rto_min, band_lo * rto_conv, recent_max_rtt)
    hi = min(rto_max, band_hi * rto_conv)
    if hi < lo:
        hi = lo
    return min(max(proposal, lo), hi, rto_max)


@dataclass
class Decision:
    rto: float                  # bounded value to use (before backoff)
    source: str                 # "ml" or "fallback"
    proposal: Optional[float]   # raw ML proposal in seconds (after margin), if any
    reason: Optional[str]       # why we fell back, if we did


class MLTimeoutEstimator:
    def __init__(self, model: Optional[ForestModel], rto_min: float, rto_max: float,
                 band_lo: float = 0.5, band_hi: float = 2.0, warmup: int = 8,
                 margin: Optional[float] = None):
        self.model = model
        self.rto_min = rto_min
        self.rto_max = rto_max
        self.band_lo = band_lo
        self.band_hi = band_hi
        self.warmup = warmup
        self.margin = margin if margin is not None else (model.margin if model else 1.0)

    def decide(self, features: Optional[List[float]], rto_conv: float,
               n_samples: int, recent_max_rtt: float) -> Decision:
        if self.model is None:
            return Decision(rto_conv, "fallback", None, "no_model")
        if features is None or n_samples < self.warmup:
            return Decision(rto_conv, "fallback", None, "warmup")
        try:
            p = self.model.predict(features)
        except Exception:                          # never let the model kill a transfer
            return Decision(rto_conv, "fallback", None, "predict_error")
        if not math.isfinite(p):
            return Decision(rto_conv, "fallback", None, "invalid_prediction")
        p = max(-PRED_LOG_CLIP, min(PRED_LOG_CLIP, p))
        if self.model.target == "correction":
            proposal = rto_conv * math.exp(p)
        else:
            proposal = math.exp(p)
        proposal *= self.margin
        if not math.isfinite(proposal) or proposal <= 0:
            return Decision(rto_conv, "fallback", None, "invalid_prediction")
        rto = apply_safety(proposal, rto_conv, recent_max_rtt, self.rto_min, self.rto_max,
                           self.band_lo, self.band_hi)
        return Decision(rto, "ml", proposal, None)


def load_estimator(path: Optional[str], rto_min: float, rto_max: float,
                   margin: Optional[float] = None) -> Tuple[MLTimeoutEstimator, Optional[str]]:
    """Load the model if possible. Returns (estimator, error message or None).

    On any error the estimator has no model and every decision falls back to
    the conventional RTO -- the transfer still works.
    """
    if not path:
        return MLTimeoutEstimator(None, rto_min, rto_max), "no model path given"
    try:
        model = ForestModel.load(path)
    except ModelError as exc:
        return MLTimeoutEstimator(None, rto_min, rto_max), str(exc)
    return MLTimeoutEstimator(model, rto_min, rto_max, margin=margin), None
