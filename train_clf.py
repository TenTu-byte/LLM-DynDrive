# -*- coding: utf-8 -*-
"""
train_remain_length_clf_auto_layer.py

Goal:
Train a classifier f(hidden_at_double_newline) -> binary(remain_length is short/long).

Pipeline (mirrors your reference script):
1) Load (hidden vec, remain_length) pairs from one or more output folders (each folder contains hidden_*.pt).
2) If --layer == -1: scan ALL COMMON layers using Ridge regression (StandardScaler + Ridge) to
   predict normalized remain_length, pick the layer with best R2 on held-out test.  (same idea as ref)  :contentReference[oaicite:3]{index=3}
3) On the best layer, build a binary label by thresholding RAW remain_length (short vs long).
   Automatically search:
   - remain_length threshold (quantiles on train)
   - probability threshold (grid on val)
   to maximize (F1, then Acc).  (same idea as ref) :contentReference[oaicite:4]{index=4}
4) Fit LogisticRegression and save:
   - <out_dir>/remain_clf.joblib
   - <out_dir>/remain_clf_meta.json  (same style as ref) :contentReference[oaicite:5]{index=5}

Notes about your hidden_*.pt format:
- expects obj["hs"] is dict[layer_id -> Tensor[K,H]]
- expects obj["remain_length"] is list/array length K
- K corresponds to number of "\\n\\n" events BEFORE </think> you decided to record.
"""

import os
import glob
import json
import time
import argparse
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
)
import joblib


# -------------------------
# utils
# -------------------------

def find_hidden_files_in_folder(folder: str) -> List[str]:
    # your outputs are typically: <folder>/hidden_{idx}.pt
    # but allow nested as well
    pat1 = os.path.join(folder, "hidden_*.pt")
    pat2 = os.path.join(folder, "**", "hidden_*.pt")
    files = sorted(glob.glob(pat1)) + sorted(glob.glob(pat2, recursive=True))
    seen = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def gather_folders(args) -> List[str]:
    folders: List[str] = []
    if isinstance(args.folder, list):
        folders.extend([x for x in args.folder if x])
    if getattr(args, "folders", None):
        folders.extend([x for x in args.folders if x])

    for k in range(1, 33):
        v = getattr(args, f"folder{k}", "") or ""
        v = v.strip()
        if v:
            folders.append(v)

    seen = set()
    out = []
    for f in folders:
        f = f.strip()
        if not f or f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


def _to_numpy_f32(x: torch.Tensor) -> np.ndarray:
    return x.detach().to(dtype=torch.float32).cpu().numpy()


def normalize_remain_length(remain: np.ndarray, mode: str = "log") -> Tuple[np.ndarray, Dict[str, float], str]:
    """
    y_norm in [0,1] for Ridge scan target.

    mode:
      - log: y = log1p(remain), then min-max
      - minmax: raw remain, then min-max
    """
    x = remain.astype(np.float32)
    meta = {}
    if mode == "log":
        x = np.log1p(x)
        label = "log1p(remain_length) min-max"
    else:
        label = "remain_length min-max"

    mn = float(x.min())
    mx = float(x.max())
    meta["y_min"] = mn
    meta["y_max"] = mx
    if mx - mn < 1e-12:
        y = np.zeros_like(x, dtype=np.float32)
    else:
        y = (x - mn) / (mx - mn)
    y = np.clip(y, 0.0, 1.0).astype(np.float32)
    return y, meta, label


def _safe_proba(clf, X: np.ndarray) -> np.ndarray:
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)
        return p[:, 1].astype(np.float32)
    if hasattr(clf, "decision_function"):
        s = clf.decision_function(X).astype(np.float32)
        return (1.0 / (1.0 + np.exp(-s))).astype(np.float32)
    return clf.predict(X).astype(np.float32)


# -------------------------
# load points (event-level)
# -------------------------

def load_hidden_meta(hidden_path: str) -> Optional[Dict[str, Any]]:
    """
    Read only what's needed to decide:
      - K (#events)
      - H (hidden dim)
      - available layers set
      - remain_length array [K]
      - hs handle
    """
    try:
        obj = torch.load(hidden_path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    hs = obj.get("hs", None)
    rem = obj.get("remain_length", None)

    if not isinstance(hs, dict) or len(hs) == 0:
        return None
    if rem is None:
        # for your new task, we REQUIRE remain_length
        return None

    remain = np.asarray(rem, dtype=np.float32)
    if remain.ndim != 1 or len(remain) <= 0:
        return None

    K = int(len(remain))

    layers = set()
    H = None
    for lid, mat in hs.items():
        try:
            lid_int = int(lid)
        except Exception:
            continue
        if torch.is_tensor(mat) and mat.ndim == 2 and int(mat.shape[0]) == K:
            if H is None:
                H = int(mat.shape[1])
                layers.add(lid_int)
            else:
                if int(mat.shape[1]) == H:
                    layers.add(lid_int)

    if H is None or len(layers) == 0:
        return None

    return {"obj": obj, "K": K, "H": H, "layers": layers, "remain": remain}


def build_event_index(
    folders: List[str],
    k: int,
    max_points: int,
    seed: int,
) -> Tuple[List[Tuple[str, int, str]], np.ndarray, int, List[int]]:
    """
    Build a list of events:
      events: List[(hidden_path, event_k, folder_tag)]
      y_rem:  np.ndarray [N]
      H: hidden dim (enforced consistent)
      common_layers: intersection across used files

    k:
      >=0 : only take that event index if exists
      -1  : take ALL event indices from each file
    """
    rng = np.random.RandomState(seed)

    all_candidates: List[Tuple[str, int, str]] = []
    file_metas: Dict[str, Dict[str, Any]] = {}

    # 1) collect candidates + per-file meta
    for folder in folders:
        tag = os.path.basename(folder.rstrip("/\\")) or folder
        for hp in find_hidden_files_in_folder(folder):
            meta = load_hidden_meta(hp)
            if meta is None:
                continue
            file_metas[hp] = meta
            K = meta["K"]
            if k >= 0:
                if k < K:
                    all_candidates.append((hp, int(k), tag))
            else:
                # all events
                for kk in range(K):
                    all_candidates.append((hp, int(kk), tag))

    if not all_candidates:
        raise RuntimeError(
            "No usable events found. Expect hidden_*.pt contains keys: 'hs' (dict[layer->Tensor[K,H]]) "
            "and 'remain_length' (len K)."
        )

    # 2) subsample events for speed/memory
    if max_points and len(all_candidates) > int(max_points):
        idx = rng.choice(len(all_candidates), size=int(max_points), replace=False)
        idx = np.asarray(idx, dtype=np.int64)
        candidates = [all_candidates[i] for i in idx.tolist()]
    else:
        candidates = all_candidates

    # 3) enforce consistent H across used files; compute common layers
    H = None
    used_files = sorted(set(hp for hp, _, _ in candidates))
    # drop files with different H
    bad = set()
    for hp in used_files:
        meta = file_metas[hp]
        if H is None:
            H = int(meta["H"])
        elif int(meta["H"]) != int(H):
            bad.add(hp)
    if bad:
        candidates = [(hp, kk, tag) for (hp, kk, tag) in candidates if hp not in bad]
        used_files = sorted(set(hp for hp, _, _ in candidates))

    if not candidates:
        raise RuntimeError("After enforcing consistent hidden dim H, no events remain.")

    common_layers = None
    for hp in used_files:
        layers = set(file_metas[hp]["layers"])
        if common_layers is None:
            common_layers = layers
        else:
            common_layers &= layers
    if common_layers is None or len(common_layers) == 0:
        raise RuntimeError("Common layer set is empty across sampled files.")

    common_layers = sorted(int(x) for x in common_layers)

    # 4) build y_rem aligned with candidates
    y_rem = np.zeros((len(candidates),), dtype=np.float32)
    for i, (hp, kk, _) in enumerate(candidates):
        y_rem[i] = float(file_metas[hp]["remain"][kk])

    return candidates, y_rem, int(H), common_layers


def load_layer_matrix_for_events(
    events: List[Tuple[str, int, str]],
    layer_id: int,
    H: int,
) -> np.ndarray:
    """
    Load X: [N,H] for given layer by iterating events (IO-heavy but memory-safe).
    """
    X = np.zeros((len(events), H), dtype=np.float32)
    # group by file for fewer loads
    by_file: Dict[str, List[Tuple[int, int]]] = defaultdict(list)  # hp -> [(row_i, kk)]
    for i, (hp, kk, _) in enumerate(events):
        by_file[hp].append((i, kk))

    for hp, pairs in by_file.items():
        obj = torch.load(hp, map_location="cpu")
        hs = obj["hs"]
        mat = hs[int(layer_id)]  # Tensor[K,H]
        for i, kk in pairs:
            X[i] = _to_numpy_f32(mat[kk])

    return X


# -------------------------
# threshold search (binary clf)
# -------------------------

def search_best_thresholds(
    X_tr: np.ndarray,
    y_rem_tr: np.ndarray,
    X_val: np.ndarray,
    y_rem_val: np.ndarray,
    seed: int,
    quantiles: List[float],
    prob_grid: np.ndarray,
    pos_is_short: bool = True,
) -> Dict[str, Any]:
    """
    For each remain_quantile q:
      thr = quantile(y_rem_tr, q)
      y_bin = (remain <= thr) if pos_is_short else (remain >= thr)
    Train LogisticRegression on X_tr, evaluate on X_val over prob_grid thresholds.
    Choose the best (F1, then Acc).
    """
    best = None
    for q in quantiles:
        thr = float(np.quantile(y_rem_tr, q))
        if pos_is_short:
            yb_tr = (y_rem_tr <= thr).astype(np.int64)
            yb_val = (y_rem_val <= thr).astype(np.int64)
        else:
            yb_tr = (y_rem_tr >= thr).astype(np.int64)
            yb_val = (y_rem_val >= thr).astype(np.int64)

        # avoid degenerate labels
        if int(yb_tr.min()) == int(yb_tr.max()):
            continue
        if int(yb_val.min()) == int(yb_val.max()):
            continue

        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=4000,
                n_jobs=-1,
                solver="lbfgs",
                random_state=seed,
            )),
        ])
        clf.fit(X_tr, yb_tr)
        p_val = _safe_proba(clf, X_val)

        for pt in prob_grid:
            pred = (p_val >= float(pt)).astype(np.int64)
            f1 = float(f1_score(yb_val, pred, average="binary"))
            acc = float(accuracy_score(yb_val, pred))
            if best is None or (f1, acc) > (best["val_f1"], best["val_acc"]):
                best = {
                    "remain_quantile": float(q),
                    "remain_threshold": thr,
                    "prob_threshold": float(pt),
                    "val_f1": f1,
                    "val_acc": acc,
                    "pos_is_short": bool(pos_is_short),
                }

    if best is None:
        raise RuntimeError("Threshold search failed (degenerate labels / too few points).")
    return best


# -------------------------
# main
# -------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--folder", type=str, action="append", default=[],
                    help="One or more folders. You can pass --folder multiple times.")
    ap.add_argument("--folders", type=str, nargs="+", default=[])
    for k in range(1, 33):
        ap.add_argument(f"--folder{k}", type=str, default="")

    ap.add_argument("--k", type=int, default=0,
                    help="Which '\\n\\n' event index to use. 0=first. -1=use ALL events.")
    ap.add_argument("--layer", type=int, required=True,
                    help="If -1, scan all common layers using Ridge R2 and pick best.")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--val_size_in_train", type=float, default=0.25)

    ap.add_argument("--max_points", type=int, default=0,
                    help="Cap #events for training (after expanding by k). 0 means no cap.")
    ap.add_argument("--ridge_alpha", type=float, default=1.0,
                    help="Ridge alpha for layer scan.")
    ap.add_argument("--norm_mode", type=str, default="log", choices=["log", "minmax"],
                    help="Normalization for Ridge scan target (remain_length).")

    ap.add_argument("--q_min", type=float, default=0.01)
    ap.add_argument("--q_max", type=float, default=0.5)
    ap.add_argument("--q_steps", type=int, default=40)
    ap.add_argument("--pos_is_short", action="store_true",
                    help="If set, y=1 means remain_length <= threshold. Otherwise y=1 means remain_length >= threshold.")

    ap.add_argument("--prob_grid_steps", type=int, default=91)

    ap.add_argument("--out_dir", type=str, default="",
                    help="Save dir for remain_clf.joblib + meta.json. Default: <first_folder>/remain_clf")

    args = ap.parse_args()

    folders = gather_folders(args)
    if not folders:
        raise SystemExit("No folder provided. Use --folder / --folders / --folder1..")

    seed = int(args.seed)
    req_layer = int(args.layer)
    k = int(args.k)

    # 1) build event index (and common layers)
    events, y_rem, H, common_layers = build_event_index(
        folders=folders,
        k=k,
        max_points=int(args.max_points),
        seed=seed,
    )
    N = len(events)
    folder_tags = np.asarray([tag for (_, _, tag) in events], dtype=object)

    y_norm, y_norm_meta, y_norm_label = normalize_remain_length(y_rem, mode=str(args.norm_mode))
    print(f"[info] events={N} | H={H} | common_layers={len(common_layers)} | k={k} | y_norm={y_norm_label}")

    # stratify by folder tag if multiple folders exist
    strat = folder_tags if len(set(folder_tags.tolist())) > 1 else None

    idx_all = np.arange(N, dtype=np.int64)
    idx_tr_all, idx_te = train_test_split(
        idx_all, test_size=float(args.test_size), random_state=seed, stratify=strat
    )

    # 2) layer scan (Ridge) if needed
    layer_scores = []
    if req_layer != -1:
        if req_layer not in common_layers:
            raise RuntimeError(f"--layer {req_layer} not in common_layers. Example available: {common_layers[:10]} ...")
        best_layer = int(req_layer)
        print(f"[info] use provided layer={best_layer} (skip scan)")
    else:
        print("\n===== Layer scan (Ridge R2 on y_norm) =====")
        best_layer = None
        best_r2 = -1e9

        for lid in common_layers:
            X = load_layer_matrix_for_events(events, layer_id=int(lid), H=H)
            X_tr = X[idx_tr_all]
            X_te = X[idx_te]
            y_tr = y_norm[idx_tr_all]
            y_te = y_norm[idx_te]

            reg = Pipeline([
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=float(args.ridge_alpha), random_state=seed)),
            ])
            reg.fit(X_tr, y_tr)
            pred = np.clip(reg.predict(X_te).astype(np.float32), 0.0, 1.0)

            r2 = float(r2_score(y_te, pred))
            mae = float(mean_absolute_error(y_te, pred))
            layer_scores.append({"layer": int(lid), "r2": r2, "mae_y_norm": mae})

            if r2 > best_r2:
                best_r2 = r2
                best_layer = int(lid)

            print(f"[scan] layer={int(lid):3d}  R2={r2:.4f}  MAE={mae:.4f}")

        layer_scores = sorted(layer_scores, key=lambda d: d["r2"], reverse=True)
        print(f"\n[best] layer={best_layer}  R2={best_r2:.4f}")

    # 3) load best layer features once
    X_best = load_layer_matrix_for_events(events, layer_id=int(best_layer), H=H)

    # split train_all -> train/val for threshold search
    idx_tr, idx_val = train_test_split(
        idx_tr_all,
        test_size=float(args.val_size_in_train),
        random_state=seed,
        stratify=(folder_tags[idx_tr_all] if strat is not None else None),
    )

    X_tr = X_best[idx_tr]
    X_val = X_best[idx_val]
    X_te = X_best[idx_te]

    y_rem_tr = y_rem[idx_tr]
    y_rem_val = y_rem[idx_val]
    y_rem_te = y_rem[idx_te]

    q_min, q_max, q_steps = float(args.q_min), float(args.q_max), max(2, int(args.q_steps))
    quantiles = np.linspace(q_min, q_max, q_steps).tolist()

    prob_steps = max(5, int(args.prob_grid_steps))
    prob_grid = np.linspace(0.05, 0.95, prob_steps).astype(np.float32)

    pos_is_short = bool(args.pos_is_short)

    print("\n===== Threshold search (remain thr + prob thr) =====")
    print(f"[search] q in [{q_min:.2f},{q_max:.2f}] x {q_steps} | prob_grid={prob_steps} | pos_is_short={pos_is_short}")

    best_cfg = search_best_thresholds(
        X_tr=X_tr, y_rem_tr=y_rem_tr,
        X_val=X_val, y_rem_val=y_rem_val,
        seed=seed,
        quantiles=quantiles,
        prob_grid=prob_grid,
        pos_is_short=pos_is_short,
    )

    rem_thr = float(best_cfg["remain_threshold"])
    prob_thr = float(best_cfg["prob_threshold"])

    # refit classifier on train_all with chosen remain threshold
    if pos_is_short:
        yb_trv = (y_rem[idx_tr_all] <= rem_thr).astype(np.int64)
        yb_te = (y_rem_te <= rem_thr).astype(np.int64)
    else:
        yb_trv = (y_rem[idx_tr_all] >= rem_thr).astype(np.int64)
        yb_te = (y_rem_te >= rem_thr).astype(np.int64)

    clf_final = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=4000, n_jobs=-1, solver="lbfgs", random_state=seed)),
    ])
    clf_final.fit(X_best[idx_tr_all], yb_trv)

    p_te = _safe_proba(clf_final, X_te)
    pred_te = (p_te >= prob_thr).astype(np.int64)

    te_f1 = float(f1_score(yb_te, pred_te, average="binary"))
    te_acc = float(accuracy_score(yb_te, pred_te))
    te_cm = confusion_matrix(yb_te, pred_te, labels=[0, 1]).astype(int)

    print("\n===== Chosen config =====")
    print(f"best_layer           : {best_layer}")
    print(f"k                   : {k}")
    print(f"remain_quantile      : {best_cfg['remain_quantile']:.3f}")
    print(f"remain_threshold     : {rem_thr:.3f} (raw remain_length)")
    print(f"prob_threshold       : {prob_thr:.3f}")
    print(f"val_f1/val_acc       : {best_cfg['val_f1']:.4f} / {best_cfg['val_acc']:.4f}")
    print(f"test_f1/test_acc     : {te_f1:.4f} / {te_acc:.4f}")
    print(f"test_cm [0/1]        :\n{te_cm}")

    out_dir = (args.out_dir.strip() or os.path.join(folders[0], "remain_clf"))
    os.makedirs(out_dir, exist_ok=True)

    joblib_path = os.path.join(out_dir, "remain_clf.joblib")
    meta_path = os.path.join(out_dir, "remain_clf_meta.json")

    joblib.dump(clf_final, joblib_path)

    meta = {
        "best_layer": int(best_layer),
        "k": int(k),
        "pos_is_short": bool(pos_is_short),
        "remain_threshold": float(rem_thr),
        "prob_threshold": float(prob_thr),
        "remain_quantile": float(best_cfg["remain_quantile"]),
        "val": {"f1": float(best_cfg["val_f1"]), "acc": float(best_cfg["val_acc"])},
        "test": {"f1": te_f1, "acc": te_acc, "cm_00_01_10_11": te_cm.reshape(-1).tolist()},
        "ridge_scan": {
            "enabled": bool(req_layer == -1),
            "ridge_alpha": float(args.ridge_alpha),
            "target_norm": y_norm_label,
            "target_norm_meta": y_norm_meta,
            "test_size": float(args.test_size),
            "scores_sorted": (sorted(layer_scores, key=lambda d: d["r2"], reverse=True) if layer_scores else []),
        },
        "data": {
            "N_events": int(N),
            "H": int(H),
            "common_layers": common_layers,
            "folders": folders,
            "max_points": int(args.max_points),
        },
        "timestamp": time.time(),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] saved classifier: {joblib_path}")
    print(f"[OK] saved meta      : {meta_path}")


if __name__ == "__main__":
    main()
