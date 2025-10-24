# parser = argparse.ArgumentParser(description="t-SNE 可视化隐藏向量，并以信心值着色")
# parser.add_argument("--layer_id", type=int, default=20)
# parser.add_argument("--jsonl_path", type=str, default="./outputs/Deepseek_7B_HiddenState/origin_temp0.7_maxlen16000.jsonl")
# parser.add_argument("--hidden_dir", type=str, default="./outputs/Deepseek_7B_HiddenState/")
# parser.add_argument("--output_png", default="./outputs/tsne_layer20_conf.png",type=str)
# parser.add_argument("--output_csv", default="", type=str, help="可选，导出(x,y,confidence)到CSV")
# parser.add_argument("--max_files", type=int, default=500)
# parser.add_argument("--expected_offset", type=int, default=1)
# parser.add_argument("--perplexity", type=float, default=30.0)
# parser.add_argument("--n_iter", type=int, default=1000)
# parser.add_argument("--random_state", type=int, default=42)
# parser.add_argument("--pca_dim", type=int, default=50,
#                     help="t-SNE前的PCA维度。<=0则跳过PCA")
# parser.add_argument("--subsample", type=int, default=0,
#                     help=">0时对总样本数下采样（均匀间隔抽样），用于加速t-SNE")
# args = parser.parse_args()
# -*- coding: utf-8 -*-
"""
t-SNE visualization for hidden vectors with confidence-based coloring.
Version with sklearn TSNE API compatibility (filters unsupported kwargs; LR auto fallback).

Usage:
python hidden_tsne_conf.py \
  --layer_id 20 \
  --jsonl_path ./outputs/DeepSeek-R1-Distill-Qwen-7B/Math_Math500/origin_temp0.7_maxlen16000.jsonl \
  --hidden_dir ./outputs/DeepSeek-R1-Distill-Qwen-7B/Math_Math500/ \
  --output_png ./outputs/DeepSeek-R1-Distill-Qwen-7B/tsne_layer20_conf.png \
  --output_csv ./outputs/DeepSeek-R1-Distill-Qwen-7B/tsne_layer20_conf.csv \
  --expected_offset 1 \
  --max_files 500 \
  --perplexity 30 \
  --n_iter 1000 \
  --pca_dim 50
"""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import json
import argparse
import inspect
from typing import List, Tuple, Optional

import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def load_hidden_for_one(
    layer_id: int,
    json_item: dict,
    hidden_path: str,
    expected_offset: int = 1,
    verbose: bool = False
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
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

    tensor = data[layer_id]  # [V, hidden_dim]


    if not torch.is_tensor(tensor) or tensor.ndim != 2:
        if verbose:
            print(f"❌ 跳过：step 形状异常，期望 2D")
        return None

    V = tensor.shape[0]
    C_raw = len(confs_raw)
    if V != C_raw - expected_offset:
        if verbose:
            print(f"❌ 跳过：V={V} 与 C_raw={C_raw} 不满足差 {expected_offset}")
        return None

    confs = confs_raw[expected_offset:][:V]
    feats_np = tensor.detach().cpu().float().numpy()
    confs_np = np.asarray(confs, dtype=np.float32)
    return feats_np, confs_np


def collect_all_features(
    layer_id: int,
    jsonl_path: str,
    hidden_dir: str,
    max_files: int = 500,
    expected_offset: int = 1,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    data_json = read_jsonl(jsonl_path)
    total = min(max_files, len(data_json))
    feats_list, confs_list = [], []
    kept_files, kept_points = 0, 0

    for i in range(total):
        hidden_path = os.path.join(hidden_dir, f"hidden_{i:03d}.pt")
        pair = load_hidden_for_one(
            layer_id=layer_id,
            json_item=data_json[i],
            hidden_path=hidden_path,
            expected_offset=expected_offset,
            verbose=True
        )
        if pair is None:
            if verbose:
                print(f"[skip] index={i}")
            continue
        feats_np, confs_np = pair
        if feats_np.shape[0] != confs_np.shape[0] or feats_np.shape[0] == 0:
            if verbose:
                print(f"[skip] index={i} 空或不齐")
            continue
        feats_list.append(feats_np)
        confs_list.append(confs_np)
        kept_files += 1
        kept_points += feats_np.shape[0]

        if (i + 1) % 50 == 0 and verbose:
            print(f"[progress] processed {i+1}/{total}, "
                  f"kept_files={kept_files}, kept_points={kept_points}")

    if not feats_list:
        raise RuntimeError("❌ 没有任何样本被成功收集，请检查 offset、layer 或输入路径。")

    X = np.concatenate(feats_list, axis=0)
    C = np.concatenate(confs_list, axis=0)
    if verbose:
        print(f"\n🎉 收集完成：X shape={X.shape}, C shape={C.shape}, "
              f"files={kept_files}/{total}, points={kept_points}")
    return X, C


def maybe_pca(X: np.ndarray, pca_dim: int, random_state: int = 42) -> np.ndarray:
    D = X.shape[1]
    if pca_dim is None or pca_dim <= 0 or pca_dim >= D:
        return X
    pca = PCA(n_components=pca_dim, random_state=random_state)
    return pca.fit_transform(X)


def build_tsne_kwargs(
    perplexity: float,
    n_iter: int,
    random_state: int,
    metric: str
) -> dict:
    """
    基于 TSNE.__init__ 的签名，过滤不可用参数；
    对 learning_rate='auto' 做后备到 200。
    """
    sig = inspect.signature(TSNE.__init__)
    allowed = set(sig.parameters.keys())

    kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "n_iter": n_iter,                # 某些版本/替代实现可能不支持
        "random_state": random_state,
        "init": "pca",
        "metric": metric,
        "learning_rate": "auto",         # 老版本不支持 'auto'
        "verbose": 1
    }
    # 只保留被支持的键
    filtered = {k: v for k, v in kwargs.items() if k in allowed}

    # learning_rate fallback
    if "learning_rate" in filtered:
        try:
            _ = TSNE(**{**filtered, "n_components": 2})
        except TypeError as e:
            # 可能是不接受 'auto'
            filtered["learning_rate"] = 200
    return filtered


def run_tsne(
    X: np.ndarray,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int = 42,
    metric: str = "euclidean",
) -> np.ndarray:
    kwargs = build_tsne_kwargs(perplexity, n_iter, random_state, metric)
    # 再次尝试构造，若仍报错则去掉 n_iter（有些第三方实现不支持）
    try:
        tsne = TSNE(**kwargs)
    except TypeError as e:
        if "n_iter" in kwargs:
            kwargs.pop("n_iter", None)
            tsne = TSNE(**kwargs)
        else:
            raise
    Y = tsne.fit_transform(X)
    return Y


def save_scatter_png(
    Y: np.ndarray,
    conf: np.ndarray,
    out_png: str,
    title: str = "t-SNE of hidden vectors",
    cmap: str = "viridis"
):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.figure(figsize=(8, 7), dpi=160)
    sc = plt.scatter(Y[:, 0], Y[:, 1], c=conf, s=4, cmap=cmap, alpha=0.9, edgecolors="none")
    cbar = plt.colorbar(sc)
    cbar.set_label("Confidence", rotation=270, labelpad=12)
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    print(f"🖼️ 已保存可视化到 {out_png}")


def maybe_save_csv(Y: np.ndarray, conf: np.ndarray, out_csv: Optional[str]):
    if not out_csv:
        return
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    arr = np.concatenate([Y, conf.reshape(-1, 1)], axis=1)
    header = "x,y,confidence"
    np.savetxt(out_csv, arr, fmt="%.6f", delimiter=",", header=header, comments="")
    print(f"📄 已保存坐标与信心到 {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="t-SNE 可视化隐藏向量，并以信心值着色")
    parser.add_argument("--layer_id", type=int, default=20)
    parser.add_argument("--jsonl_path", type=str, default="./outputs/Deepseek_7B_HiddenState/origin_temp0.7_maxlen16000.jsonl")
    parser.add_argument("--hidden_dir", type=str, default="./outputs/Deepseek_7B_HiddenState/")
    parser.add_argument("--output_png", default="./outputs/tsne_layer35_conf_14b.png",type=str)
    parser.add_argument("--output_csv", default="", type=str, help="可选，导出(x,y,confidence)到CSV")
    parser.add_argument("--max_files", type=int, default=500)
    parser.add_argument("--expected_offset", type=int, default=1)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--pca_dim", type=int, default=50,
                         help="t-SNE前的PCA维度。<=0则跳过PCA")
    parser.add_argument("--subsample", type=int, default=0,
                         help=">0时对总样本数下采样（均匀间隔抽样），用于加速t-SNE")
    args = parser.parse_args()


    X, C = collect_all_features(
        layer_id=args.layer_id,
        jsonl_path=args.jsonl_path,
        hidden_dir=args.hidden_dir,
        max_files=args.max_files,
        expected_offset=args.expected_offset,
        verbose=True
    )

    if args.subsample and args.subsample > 0 and args.subsample < X.shape[0]:
        step = max(1, X.shape[0] // args.subsample)
        idx = np.arange(0, X.shape[0], step, dtype=int)[:args.subsample]
        X = X[idx]
        C = C[idx]
        print(f"🔎 下采样: 选取 {len(idx)} 条样本用于 t-SNE")

    Xp = maybe_pca(X, args.pca_dim, random_state=args.random_state)
    if Xp is not X:
        print(f"🧪 PCA: {X.shape[1]} -> {Xp.shape[1]}")

    Y = run_tsne(
        Xp,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        random_state=args.random_state,
        metric="euclidean",
    )

    title = f"t-SNE (layer={args.layer_id}) colored by confidence"
    save_scatter_png(Y, C, args.output_png, title=title, cmap="viridis")
    maybe_save_csv(Y, C, args.output_csv)


if __name__ == "__main__":
    main()
