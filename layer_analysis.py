# -*- coding: utf-8 -*-
"""
Evaluate per-layer correlation with confidence using PCA + Ridge.
Metrics: R^2 (on test set) and Spearman correlation (ρ).

Usage example:
python hidden_conf_ridge.py \
  --jsonl_path ./outputs/DeepSeek-R1-Distill-Qwen-7B/Math_Math500/origin_temp0.7_maxlen16000.jsonl \
  --hidden_dir ./outputs/DeepSeek-R1-Distill-Qwen-7B/Math_Math500/ \
  --layers 0-28 \
  --max_files 150 \
  --expected_offset 1 \
  --alpha 1.0 \
  --pca_components 64 \
  --test_size 0.2 \
  --random_state 42
"""
import os
import re
import json
import math
import argparse
from typing import Optional, Tuple, List

import numpy as np
#import torch
import torch
from torch.utils.data import TensorDataset, ConcatDataset

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.stats import spearmanr

# --- 可选依赖：SciPy 的 spearmanr（若没有会自动用无依赖实现） ---
try:
    from scipy.stats import spearmanr as _scipy_spearmanr
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ----------------------
# IO helpers
# ----------------------
def read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


# ----------------------
# Spearman ρ（优先 SciPy，缺省则用无依赖实现）
# ----------------------
def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    返回 (rho, p_value)。若无 SciPy，则 p_value 返回 NaN。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if _HAS_SCIPY:
        rho, p = _scipy_spearmanr(y_true, y_pred)
        return float(rho), float(p)

    # ---- 无 SciPy 的备用实现：平均名次法 + 皮尔逊相关 ----
    def _average_ranks(a: np.ndarray) -> np.ndarray:
        idx = np.argsort(a, kind="mergesort")  # 稳定排序，便于处理并列
        ranks = np.empty_like(idx, dtype=float)
        sorted_a = a[idx]
        n = len(a)
        start = 0
        for i in range(n):
            # 分组：相等的一段作为一组
            if i == n - 1 or sorted_a[i] != sorted_a[i + 1]:
                end = i
                # 平均名次（1..n）
                avg_rank = (start + end) / 2.0 + 1.0
                ranks[idx[start:end + 1]] = avg_rank
                start = i + 1
        return ranks

    rx = _average_ranks(y_true)
    ry = _average_ranks(y_pred)
    rho = np.corrcoef(rx, ry)[0, 1]
    return float(rho), float("nan")


# ----------------------
# Single-file loader (one hidden_i.pt)
# Aligns 'step' vectors with sentence_confidences using expected_offset
# Returns torch TensorDataset(features, label=confidence)
# ----------------------
def build_dataset_from_layer_conf(
    layer_id: int,
    json_item: dict,
    hidden_path: str,
    expected_offset: int = 1,
    verbose: bool = False
) -> Optional[TensorDataset]:
    """
    For a single sample (hidden_i.pt + one json item):
      - Load layer's step hidden vectors: [V, D]
      - Take confidences from sentence_confidences[expected_offset : expected_offset+V]
      - Return TensorDataset(features, labels)
    Skip and return None if shape or files don't match.
    """
    confs_raw = json_item.get("sentence_confidences", [])
    if not isinstance(confs_raw, list) or len(confs_raw) <= expected_offset:
        if verbose:
            print("⛔ 跳过：sentence_confidences 为空或长度不足")
        return None

    if not os.path.exists(hidden_path):
        if verbose:
            print(f"❌ 跳过：hidden 文件不存在 {hidden_path}")
        return None

    try:
        data = torch.load(hidden_path, map_location="cpu", weights_only=False)
    except Exception as e:
        if verbose:
            print(f"❌ 加载失败 {hidden_path}: {e}")
        return None

    if layer_id not in data:
        if verbose:
            print(f"❌ 跳过：layer {layer_id} 不在 {os.path.basename(hidden_path)} 中")
        return None

    # ! strict the dict structure
    tensor = data[layer_id]  # [V, D]
    if not torch.is_tensor(tensor) or tensor.ndim != 2:
        if verbose:
            print(f"❌ 跳过：step 形状异常，期望 2D 张量")
        return None

    V = tensor.shape[0]
    C_raw = len(confs_raw)
    if V != C_raw - expected_offset:
        if verbose:
            print(f"❌ 跳过：V={V} 与 C_raw={C_raw} 不满足差 {expected_offset}")
        return None

    confs = confs_raw[expected_offset: expected_offset + V]
    if len(confs) != V:
        if verbose:
            print("❌ 跳过：对齐后置信度长度不等于 V")
        return None

    feats = tensor.to(dtype=torch.float32)
    labels = torch.tensor(confs, dtype=torch.float32)
    return TensorDataset(feats, labels)


# ----------------------
# Batch builder across many files hidden_0.pt ... hidden_{N-1}.pt
# Returns ConcatDataset of (feature, confidence)
# ----------------------
def batch_build_all(
    layer_id: int,
    jsonl_path: str,
    hidden_dir: str,
    max_files: int = 150,
    expected_offset: int = 1,
    file_pattern: str = "hidden_{idx}.pt",
    verbose: bool = True
) -> ConcatDataset:
    data_json = read_jsonl(jsonl_path)
    total = min(max_files, len(data_json))
    datasets = []
    kept_samples, kept_files = 0, 0

    for i in range(total):
        hidden_path = os.path.join(hidden_dir, file_pattern.format(idx=i))
        ds = build_dataset_from_layer_conf(
            layer_id=layer_id,
            json_item=data_json[i],
            hidden_path=hidden_path,
            expected_offset=expected_offset,
            verbose=True
        )
        if ds is not None:
            datasets.append(ds)
            kept_files += 1
            kept_samples += len(ds)
        else:
            if verbose:
                print(f"[skip] index={i}")

        if (i + 1) % 50 == 0 and verbose:
            print(f"[progress] processed {i+1}/{total}, "
                  f"kept_files={kept_files}, kept_samples={kept_samples}")

    if not datasets:
        raise RuntimeError("❌ 没有任何样本被成功收集，请检查路径/层号/offset。")

    merged = ConcatDataset(datasets)
    if verbose:  # ! actually, step_num correspond to sample_num, and sample_num correspond to file_num
        print(f"\n🎉 合并完成：样本数={len(merged)}，文件数={kept_files}/{total}")
    return merged


# ----------------------
# Evaluate PCA + Ridge → R^2 + Spearman ρ
# ----------------------
def evaluate_pca_ridge(merged_dataset: ConcatDataset,
                       alpha: float = 1.0,
                       test_size: float = 0.2,
                       random_state: int = 42,
                       pca_components: int = 64) -> Tuple[float, float]:
    all_features = []
    all_labels = []

    for i in range(len(merged_dataset)):  # step_num
        feature, label = merged_dataset[i]  # feature: [D], label: confidence scalar
        # 转 numpy
        all_features.append(feature.to(torch.float32).numpy())
        all_labels.append(float(label.item()))

    X = np.asarray(all_features, dtype=np.float32)  # [N, D]
    y = np.asarray(all_labels, dtype=np.float32)    # [N]

    # 拆分训练/测试
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(np.shape(X_train))

    # PCA 降维
    max_comp = min(pca_components, X_train.shape[0], X_train.shape[1])
    if max_comp <= 0:
        raise ValueError("PCA 组件数无效，请检查样本量或设置更小的 --pca_components")
    pca = PCA(n_components=max_comp, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # === 新增：输出保留的信息量（累计解释方差） ===
    retained_var = float(np.sum(pca.explained_variance_ratio_))  # ∈ (0, 1]
    print(f"[PCA] k={max_comp}  retained variance = {retained_var:.4f}  ({retained_var * 100:.2f}%)")

    # 可选：顺便打印达到常见阈值所需的 k
    cum = np.cumsum(pca.explained_variance_ratio_)
    for t in (0.80, 0.90, 0.95, 0.99):
        k = int(np.searchsorted(cum, t) + 1)
        if k <= len(cum):
            print(f"[PCA] retain ≥ {t:.0%}  → k = {k}")

    # Ridge 回归
    model = Ridge(alpha=alpha)  # min​J(w)=∣∣y−X_w∣∣**2+α∣∣w∣∣**2
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)

    # R^2
    r2 = r2_score(y_test, y_pred)  # ! focus on the value

    # Spearman ρ（基于测试集的 y_true vs y_pred）
    rho, pval = spearmanr(y_test, y_pred)  # ! focus on the rank

    # 输出
    if math.isnan(pval):
        print(f"📊 PCA+Ridge R² = {r2:.4f} | Spearman ρ = {rho:.4f}（components={max_comp}, alpha={alpha}）")
    else:
        print(f"📊 PCA+Ridge R² = {r2:.4f} | Spearman ρ = {rho:.4f} (p={pval:.2e})（components={max_comp}, alpha={alpha}）")

    return float(r2), float(rho)


# ----------------------
# Layer range parser
# ----------------------
def parse_layers(layer_str: str) -> List[int]:
    """
    支持:
      "20" -> [20]
      "0-28" -> [0,1,...,28]
      "1,5,7-10" -> [1,5,7,8,9,10]
    """
    layers: List[int] = []
    parts = re.split(r"[,\s]+", layer_str.strip())
    for p in parts:
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)  # maxsplit = 1
            a, b = int(a), int(b)
            if a <= b:
                layers.extend(range(a, b + 1))
            else:
                layers.extend(range(a, b - 1, -1))
        else:
            layers.append(int(p))
    # 去重且按序
    return sorted(set(layers))


# ----------------------
# Main
# ----------------------
def main():
    ap = argparse.ArgumentParser(description="Per-layer confidence regression via PCA+Ridge (R² & Spearman ρ)")
    ap.add_argument("--jsonl_path", type=str, default="./outputs/Qwen3_14B_HiddenState/origin_temp0.7_maxlen16000.merged.jsonl")
    ap.add_argument("--hidden_dir", type=str, default="./outputs/Qwen3_14B_HiddenState/")
    ap.add_argument("--layers", type=str, default="28-30", help="如 0-28 或 0,3,7-10")
    ap.add_argument("--max_files", type=int, default=500)
    ap.add_argument("--expected_offset", type=int, default=1)
    ap.add_argument("--file_pattern", type=str, default="hidden_{idx}.pt")

    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--pca_components", type=int, default=64)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)

    args = ap.parse_args()
    layer_ids = parse_layers(args.layers)

    metrics: List[Tuple[int, Optional[float], Optional[float]]] = []

    for layer in layer_ids:
        print(f"\n📂 正在处理 Layer {layer} ...")
        try:
            merged_dataset = batch_build_all(
                layer_id=layer,
                jsonl_path=args.jsonl_path,
                hidden_dir=args.hidden_dir,
                max_files=args.max_files,
                expected_offset=args.expected_offset,
                file_pattern=args.file_pattern,
                verbose=True
            )
            r2, rho = evaluate_pca_ridge(
                merged_dataset,
                alpha=args.alpha,
                pca_components=args.pca_components,
                test_size=args.test_size,
                random_state=args.random_state
            )
            metrics.append((layer, r2, rho))
            print(f"✅ Layer {layer} 测试集 R² = {r2:.4f} | Spearman ρ = {rho:.4f}")
        except Exception as e:
            print(f"❌ Layer {layer} 失败：{e}")
            metrics.append((layer, None, None))

    # 汇总输出

    print("\n🎯 各层 PCA+Ridge 测试集指标：")
    for layer, r2, rho in metrics:

        if (r2 is not None) and (rho is not None):

            print(f"Layer {layer:2d}: R² = {r2:.4f} | Spearman ρ = {rho:.4f}")
        else:
            print(f"Layer {layer:2d}: ❌ 无结果")


if __name__ == "__main__":
    main()

