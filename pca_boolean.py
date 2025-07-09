#!/usr/bin/env python3
"""
PCA comparison for cached **ManyPaths** tasks.

✔ Handles **both** modern caches that end with a metadata *dict* **and** the
  earlier 9‑tuple format that stores only tensors + literal/depth counts.
✔ Chooses one representative task from each literal bucket (Simple, Medium,
  Complex) and draws PC1–PC2 scatter plots (teal = positive, grey = negative).
✔ If the task does **not** include the expression tree, PCA is run on the
  *observed* inputs (support ✚ query) rather than the full 2^8 cube.
"""
from __future__ import annotations

import argparse, pathlib, random, sys
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import torch

plt.rcParams["font.family"] = ["Helvetica", "DejaVu Sans", "sans-serif"]
plt.rcParams["font.size"] = 8
TEAL, GRAY = "#0F9D9D", "#CCCCCC"

# ────────────────────────────────────────────────────────────────
#  Cache loading utilities
# ────────────────────────────────────────────────────────────────

def load_tasks(path: pathlib.Path) -> List[Any]:
    """Return list of task objects from a cache *.pt* file."""
    obj = torch.load(path, map_location="cpu")
    # ManyPaths often saves (tasks, meta_info)
    if (isinstance(obj, (list, tuple)) and len(obj) == 2
            and isinstance(obj[0], Sequence)):
        obj = obj[0]
    if not isinstance(obj, Sequence):
        sys.exit("❌  Cache file does not contain a sequence of tasks.")
    return list(obj)

# ────────────────────────────────────────────────────────────────
#  Meta‑dict extraction (robust to old/new formats)
# ────────────────────────────────────────────────────────────────

def meta_dict(task: Any) -> Dict[str, Any]:
    """Return a metadata dict with keys literals, depth, num_features, expr."""
    # 1) New format: last element is dict
    if isinstance(task, (tuple, list)) and isinstance(task[-1], dict):
        md = task[-1].copy()
        md.setdefault("expr", None)
        md.setdefault("num_features", 8)
        return md
    # 2) Old 9‑tuple format
    if isinstance(task, (tuple, list)) and len(task) >= 9:
        # tuple indices: …, n_support, literals, depth
        return {
            "literals": int(task[-2]),
            "depth": int(task[-1]),
            "expr": None,
            "num_features": task[1].shape[1] if hasattr(task[1], "shape") else 8,
            "_old_tuple": True
        }
    # 3) Task is already dict‑like
    if isinstance(task, dict):
        md = task.copy(); md.setdefault("expr", None); md.setdefault("num_features", 8); return md
    raise ValueError("Unrecognised task format → cannot extract metadata")

# ────────────────────────────────────────────────────────────────
#  Convert *one* task to (X, y) matrix + meta
# ────────────────────────────────────────────────────────────────

def task_to_xy(task: Any) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    md = meta_dict(task)
    n = md["num_features"]

    # Case A: full expression available → evaluate on entire Boolean cube
    if md["expr"] is not None:
        from pcfg import evaluate_pcfg_concept  # local import to avoid heavy dep if unused
        X_full = np.array([[ (i >> k) & 1 for k in range(n) ] for i in range(1 << n)], float)
        y_full = np.array([ evaluate_pcfg_concept(md["expr"], x) for x in X_full ], bool)
        return X_full, y_full, md

    # Case B: old tuple → use support+query originals only
    if md.get("_old_tuple"):
        X_sup = task[1].cpu().numpy()  # original support vectors
        y_sup = task[2].cpu().numpy().ravel() > 0.5
        X_q   = task[4].cpu().numpy()  # original query vectors
        y_q   = task[5].cpu().numpy().ravel() > 0.5
        X = np.vstack([X_sup, X_q]).astype(float)
        y = np.hstack([y_sup, y_q]).astype(bool)
        return X, y, md

    raise RuntimeError("Could not derive X,y for task – unknown format")

# ────────────────────────────────────────────────────────────────
#  Bucket selection (Simple / Medium / Complex)
# ────────────────────────────────────────────────────────────────

def representative_tasks(tasks: List[Any]) -> Dict[str, Any]:
    buckets = {"Simple": (1,3), "Medium": (4,6), "Complex": (7, 1000)}
    rng = random.Random(0)
    reps = {}
    for name, (lo, hi) in buckets.items():
        cand = [t for t in tasks if lo <= meta_dict(t)["literals"] <= hi]
        if cand:
            reps[name] = rng.choice(cand)
        else:
            print(f"⚠️  No {name} task (literals {lo}-{hi}) found.")
    return reps

# ────────────────────────────────────────────────────────────────
#  Plotting
# ────────────────────────────────────────────────────────────────

def plot_pca_panels(reps: Dict[str, Any], outfile="concept_pca_panels.svg"):
    fig, axes = plt.subplots(1, 3, figsize=(12,4), dpi=160, sharex=True, sharey=True)

    for ax, (bucket, task) in zip(axes, reps.items()):
        X, y, md = task_to_xy(task)
        pcs = PCA(n_components=2, random_state=0).fit_transform(X - X.mean(0))
        ax.scatter(*pcs[y].T,  c=TEAL, s=20, alpha=.85, label="Positive")
        ax.scatter(*pcs[~y].T, c=GRAY, s=20, alpha=.25, label="Negative")
        ax.set_title(f"{bucket}\n({md['literals']} lits, depth {md['depth']})", fontweight="bold", pad=8)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_xticks([]); ax.set_yticks([])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=7, frameon=False)
    fig.suptitle("PCA Projection of Cached ManyPaths Concepts", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0,0,0.97,0.92])
    fig.savefig(outfile)
    print(f"✅  Figure saved → {outfile}")
    plt.show()

# ────────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Draw PCA panels for a ManyPaths cache")
    ap.add_argument("cache", type=pathlib.Path, help="Path to .pt cache file")
    args = ap.parse_args()

    tasks = load_tasks(args.cache)
    reps  = representative_tasks(tasks)
    if not reps:
        sys.exit("❌  No tasks matched the literal buckets – adjust ranges or check cache.")
    plot_pca_panels(reps, outfile=args.cache.with_suffix("_pca.svg"))

if __name__ == "__main__":
    main()
