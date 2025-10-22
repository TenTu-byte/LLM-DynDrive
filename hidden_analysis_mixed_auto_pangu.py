# -*- coding: utf-8 -*-
"""
hidden_analysis_mixed_auto.py

在原版基础上，保留：
  - S = mean(H_hit) - mean(H_non) 的保存
  - “μ_hit → H_non”与“μ_non → H_hit”的最短/最长欧氏距离统计

新增（命令行不变即可运行；新增参数均有默认值）：
  - 对角近似 LDA 分割，计算：
      * 沿 -u_w 的最短覆盖距离 d_all_w
      * 固定沿 -S 的最短覆盖距离 d_all_S 与系数 alpha_all_S
      * 仅均值越界的 d_mean_w, d_mean_S, alpha_mean_S
  - 提取 jsonl 中所有信心值并计算分位数 q25 / q75
  - 命令行兼容保留 --gamma/--epsilon/--delta，但不再用于“安全余量”运算
"""

import os
import json
import argparse
from typing import Tuple

import torch
import torch_npu
from torch.utils.data import ConcatDataset

# ===== 你的工程内已有的构建数据集函数 =====
try:
    from hidden_analysis_mixed_pangu import batch_build_all_mixed
except Exception as e:
    raise ImportError(
        "无法从 hidden_analysis_mixed_pangu 导入 batch_build_all_mixed。请确认该文件与本脚本同目录或在 PYTHONPATH 中。"
        f" 原始错误：{repr(e)}"
    )


def compute_confidence_quartiles(jsonl_path: str):
    # 从 JSONL 中尽可能鲁棒地抽取所有“信心值”，返回 (q25, q75, N)。
    # 支持键：
    #   - 直接键： "confidence", "conf", "score"
    #   - 列表键： "sentence_confidences", "confidences", "scores", "sentence_scores", "step_confidences"
    #   - 任意层级嵌套时，若元素是 dict 并含有上述直接键，也会被抽取
    import json, math
    import torch

    def _collect_from_obj(o, sink):
        if isinstance(o, dict):
            for k, v in o.items():
                lk = k.lower()
                # 直接数值
                if lk in ("confidence", "conf", "score") and isinstance(v, (int, float)) and math.isfinite(v):
                    sink.append(float(v))
                # 列表字段
                elif lk in ("sentence_confidences", "confidences", "scores", "sentence_scores", "step_confidences"):
                    if isinstance(v, (list, tuple)):
                        for x in v:
                            if isinstance(x, (int, float)) and math.isfinite(x):
                                sink.append(float(x))
                            elif isinstance(x, dict):
                                for kk in ("confidence", "conf", "score"):
                                    if kk in x and isinstance(x[kk], (int, float)) and math.isfinite(x[kk]):
                                        sink.append(float(x[kk]))
                # 递归遍历其他嵌套
                if isinstance(v, (dict, list, tuple)):
                    _collect_from_obj(v, sink)
        elif isinstance(o, (list, tuple)):
            for x in o:
                _collect_from_obj(x, sink)

    confidences = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                _collect_from_obj(obj, confidences)
    except FileNotFoundError:
        return None, None, 0

    if not confidences:
        return None, None, 0

    t = torch.tensor(confidences, dtype=torch.float64)
    q25 = torch.quantile(t, 0.25).item()
    q75 = torch.quantile(t, 0.75).item()
    return q25, q75, len(confidences)


# ====== 工具：收集两类向量 ======
def _collect_vectors_by_class(merged_dataset: ConcatDataset) -> Tuple[torch.Tensor, torch.Tensor]:
    """从合并数据集收集两类向量，返回 H_hit, H_non（均为 [N, D] 的 float32 Tensor）"""
    hits, nonhits = [], []
    for i in range(len(merged_dataset)):
        feat, y = merged_dataset[i]
        feat = feat.view(-1).to(torch.float32)  # 保证 1D
        label = int(y.item()) if hasattr(y, "item") else int(y)
        if label == 1:
            hits.append(feat)
        else:
            nonhits.append(feat)
    if len(hits) == 0 or len(nonhits) == 0:
        raise ValueError(f"类别不平衡：hit={len(hits)}, non={len(nonhits)}")
    H_hit = torch.stack(hits, dim=0)   # [N1, D]
    H_non = torch.stack(nonhits, dim=0)  # [N0, D]
    return H_hit, H_non


# ====== 工具：中心→集合最短/最长距离（分块） ======
@torch.no_grad()
def center_to_set_minmax(center: torch.Tensor,
                         H: torch.Tensor,
                         block_size: int = 65536,
                         device: str = "cpu") -> Tuple[float, float]:
    """
    计算“中心向量 center 到集合 H 的最短/最长欧氏距离”，分块以节省显存/内存。
    center: [D], H: [N, D]
    返回: (min_dist, max_dist)
    """
    assert center.dim() == 1 and H.dim() == 2 and center.size(0) == H.size(1), \
        f"维度不匹配: center {center.shape}, H {H.shape}"
    center = center.to(torch.float32).to(device)
    H = H.to(torch.float32).to(device)

    n = H.size(0)
    min_d = float("inf")
    max_d = 0.0
    for s in range(0, n, block_size):
        e = min(s + block_size, n)
        block = H[s:e]  # [B, D]
        diff = block - center  # [B, D]
        d = torch.linalg.norm(diff, dim=1)  # [B]
        min_d = min(min_d, float(d.min().item()))
        max_d = max(max_d, float(d.max().item()))
    return min_d, max_d


# ====== 核心：构建中心与范围 ======
@torch.no_grad()
def build_centers_and_ranges(merged_dataset: ConcatDataset,
                             center_block_size: int = 65536,
                             device: str = "cpu"):
    """
    从合并数据集中构造：
      - S = mean(H_hit) - mean(H_non)
      - H_hit 与 H_non 两个类的中心 μ_hit, μ_non
      - μ_hit → H_non 的最短/最长距离；μ_non → H_hit 的最短/最长距离
    """
    H_hit, H_non = _collect_vectors_by_class(merged_dataset)  # [N1, D], [N0, D]

    # 计算中心
    mu_hit = H_hit.mean(dim=0)  # [D]
    mu_non = H_non.mean(dim=0)  # [D]
    S = (mu_hit - mu_non).to(torch.float32)  # [D]

    # 距离范围（分块）
    hit_to_non_min, hit_to_non_max = center_to_set_minmax(mu_hit, H_non, center_block_size, device)
    non_to_hit_min, non_to_hit_max = center_to_set_minmax(mu_non, H_hit, center_block_size, device)

    return S, mu_hit, mu_non, H_hit, H_non, hit_to_non_min, hit_to_non_max, non_to_hit_min, non_to_hit_max


# ====== LDA（对角近似）与 steer 计算 ======
@torch.no_grad()
def compute_lda_separator_and_steer(merged_dataset: ConcatDataset, ridge: float = 0.0):
    """
    使用“对角协方差近似”的 LDA，得到分割向量 w 与阈值 b，使得 sign(w·x + b) 作为线性判别。
    仅用于分析/方向估计，不做分类训练。
    """
    H_hit, H_non = _collect_vectors_by_class(merged_dataset)
    X1 = H_hit.to(torch.float64)
    X0 = H_non.to(torch.float64)

    mu1 = X1.mean(dim=0)
    mu0 = X0.mean(dim=0)
    S1 = X1.var(dim=0, unbiased=False)  # (1/N)
    S0 = X0.var(dim=0, unbiased=False)

    # 加微小 ridge，避免极小方差导致数值不稳
    if ridge > 0:  # TODO: not used?
        S1 = S1 + ridge
        S0 = S0 + ridge

    # 对角近似 LDA：w = (mu1 - mu0) / (S1 + S0)
    denom = S1 + S0 + 1e-12
    w = (mu1 - mu0) / denom
    w = w.to(torch.float64)

    # 判别阈值 b，采用 Fisher 判别的均值中点（简化处理）
    # 目标：w·x + b > 0 归为 1 类
    m1 = (w * mu1).sum().item()
    m0 = (w * mu0).sum().item()
    t = -0.5 * (m1 + m0)  # 阈值 t = -b
    b = -t

    # 一些方向范数
    S_vec = (mu1 - mu0)
    w_norm = float(torch.linalg.norm(w).item())
    S_norm = float(torch.linalg.norm(S_vec).item())
    if w_norm < 1e-12 or S_norm < 1e-12:
        return {"ok": False, "reason": "方向范数过小"}

    # 沿 -u_w (w方向的单位向量) 的“全体越界”最短距离（使所有 hit 样本越过判别面）
    # 对每个 hit 样本 h：需要 d_i 使 t - w·h + d_i * ||w|| >= 0
    # => d_i >= (w·h - t) / ||w||
    proj_hit = (X1 @ w).double()  # [N1]
    req = (proj_hit - t) / (w_norm + 1e-12)  # [N1]
    d_all_w = float(req.max().item())  # TODO: computed values are multiples of ||w||, not actual distance?

    # 固定沿 -S 方向（单位向量 u_S）
    uS = S_vec / (S_norm + 1e-12)  # TODO: why normalized first, not projecting directly?
    w_dot_uS = float((w * uS).sum().item())  # w·uS
    if w_dot_uS <= 0:
        # 若 w 与 S 方向夹角过大，沿 -S 无法降低判别值
        d_all_S = float("inf")
        alpha_all_S = float("inf")
        d_mean_S = float("inf")
        alpha_mean_S = float("inf")
        d_mean_w = max((m1 - t) / (w_norm + 1e-12), 0.0)
    else:
        # 使所有 hit 越界所需投影距离（沿 -uS 的欧氏距离）：max_i (w·h_i - t) / (w·uS)
        # TODO: same as above; multiples, not distance
        alpha_u_all = float(((proj_hit - t) / (w_dot_uS + 1e-12)).max().item())  # TODO: are proj_his and uS mapped onto or alonw w?
        # 将“沿 -uS 的距离”换算为“沿 -S（非单位向量）的系数”：alpha_S = alpha_u / ||S||
        alpha_all_S = alpha_u_all / (S_norm + 1e-12)
        # Euclidean 距离沿 -uS 的最短移动量
        d_all_S = alpha_u_all

        # 仅均值越界：mu1
        d_mean_w = max((m1 - t) / (w_norm + 1e-12), 0.0)  # TODO: same as above; multiples, not distance
        d_mean_S_u = max((m1 - t) / (w_dot_uS + 1e-12), 0.0)   # 沿 -uS 的欧氏距离
        alpha_mean_S = d_mean_S_u / (S_norm + 1e-12)           # 换算为沿 -S 的系数
        d_mean_S = d_mean_S_u

    return {
        "ok": True,
        "w_norm": w_norm,
        "S_norm": S_norm,
        "w": w,
        "S": S_vec.to(torch.float64),
        "b": b,
        "threshold_t": t,
        "max_w_dot_h": float(proj_hit.max().item()),
        "d_all_w": d_all_w,
        "d_all_S": d_all_S,
        "alpha_all_S": alpha_all_S,
        "d_mean_w": float((m1 - t) / (w_norm + 1e-12) if w_norm > 0 else float("inf")),
        "d_mean_S": d_mean_S,
        "alpha_mean_S": alpha_mean_S,
        "w_dot_uS": w_dot_uS,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compute S, center-to-other distance ranges, and LDA-based steer thresholds (command unchanged)"
    )
    parser.add_argument("--layer_id", type=int, required=True)
    parser.add_argument("--jsonl_path", type=str, required=True)
    parser.add_argument("--hidden_dir", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True,
                        help="保存均值差向量 S 的 .pt 路径")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--max_files", type=int, default=100)
    parser.add_argument("--expected_offset", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")

    # 兼容旧参数（忽略不用）
    parser.add_argument("--pairwise_block_rows", type=int, default=2048, help="[兼容参数] 已忽略")
    parser.add_argument("--pairwise_block_cols", type=int, default=4096, help="[兼容参数] 已忽略")

    # 有效参数（与原脚本一致）
    parser.add_argument("--center_block_size", type=int, default=65536,
                        help="中心到集合距离的分块大小（一次处理的向量条数）")
    parser.add_argument("--device", type=str, choices=["cpu", "npu"], default="cpu")
    parser.add_argument("--report_path", type=str, default="",
                        help="可选：写出 JSON 报告到该路径")

    # 为保持命令行兼容，以下三个参数仍保留，但不再参与任何“安全余量”计算
    parser.add_argument("--gamma", type=float, default=1.5, help="[兼容保留]")
    parser.add_argument("--epsilon", type=float, default=150, help="[兼容保留]")
    parser.add_argument("--delta", type=float, default=35, help="[兼容保留]")

    args = parser.parse_args()

    # 构建 (feat, label) 合并数据集
    merged = batch_build_all_mixed(  # TODO: how to determine the layer_id?
        layer_id=args.layer_id,
        jsonl_path=args.jsonl_path,
        hidden_dir=args.hidden_dir,
        threshold=args.threshold,
        max_files=args.max_files,
        expected_offset=args.expected_offset,
        verbose=args.verbose
    )

    (S, mu_hit, mu_non,
     H_hit, H_non,
     hit_to_non_min, hit_to_non_max,
     non_to_hit_min, non_to_hit_max) = build_centers_and_ranges(
        merged,
        center_block_size=args.center_block_size,
        device=args.device
    )

    # 保存 S
    if os.path.dirname(args.save_path):
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(S.cpu(), args.save_path)

    # 打印中心→集合距离与 S
    dim = S.numel()
    l2 = float(torch.linalg.norm(S).item())
    print(f"✅ 已保存 steer 向量到 {args.save_path}")
    print(f"   dim={dim} | ||S||_2 = {l2:.6f} | hit={H_hit.size(0)} | nonhit={H_non.size(0)}")
    print(f"   μ_hit → H_non  最短距离: {hit_to_non_min:.6f} | 最长距离: {hit_to_non_max:.6f}")
    print(f"   μ_non → H_hit  最短距离: {non_to_hit_min:.6f} | 最长距离: {non_to_hit_max:.6f}")

    # ===== LDA 分割与 steer（ridge=0，无先验；保留原有核心指标） =====
    lda = compute_lda_separator_and_steer(merged)
    if lda.get("ok", False):
        print("\n—— LDA (对角) 分割与 steer ——")
        print(f"  • ||w||_2 = {lda['w_norm']:.6f} | ||S||_2 = {lda['S_norm']:.6f}")
        print(f"  • 判别阈值 t = -b = {lda['threshold_t']:.6f} | max_i(w·h_i) = {lda['max_w_dot_h']:.6f}")

        print("  • 最优方向（沿 -u_w）:")
        print(f"      d_all_w  = {lda['d_all_w']:.6f}   # 使所有 hit 进入 non 区域的最短欧氏距离")

        print("  • 固定你的方向（沿 -S）:")
        if lda['d_all_S'] == float('inf'):
            print("      与 w 方向夹角过大，沿 -S 无法降低判别值；请改用 -w 或检查 S 的定义。")
        else:
            print(f"      d_all_S  = {lda['d_all_S']:.6f}   | alpha_all_S = {lda['alpha_all_S']:.6f}")

        print("  • 仅均值越界：")
        if lda['d_mean_S'] == float('inf'):
            print(f"      d_mean_w = {lda['d_mean_w']:.6f} | d_mean_S 无法定义（w·S ≤ 0）")
        else:
            print(f"      d_mean_w = {lda['d_mean_w']:.6f} | d_mean_S = {lda['d_mean_S']:.6f} | alpha_mean_S = {lda['alpha_mean_S']:.6f}")
    else:
        print(f"⚠️ LDA 分割失败：{lda.get('reason','unknown')}")

    # ===== 统计所有“信心值”的 q25 / q75 并打印 =====
    q25, q75, n_conf = compute_confidence_quartiles(args.jsonl_path)
    if n_conf > 0:
        print(f"   confidence 分位数: q25 = {q25:.6f} | q75 = {q75:.6f} | N = {n_conf}")
    else:
        print("   未从 JSONL 提取到任何 confidence 值；跳过分位数统计。")

    # ===== 可选报告落盘 =====
    if args.report_path:
        report = {
            "layer_id": args.layer_id,
            "dim": dim,
            "n_hit": H_hit.size(0),
            "n_non": H_non.size(0),
            "S_l2": l2,
            "hit_to_non_min": hit_to_non_min,
            "hit_to_non_max": hit_to_non_max,
            "non_to_hit_min": non_to_hit_min,
            "non_to_hit_max": non_to_hit_max,
            "jsonl_path": args.jsonl_path,
            "hidden_dir": args.hidden_dir,
            "threshold": args.threshold,
            "center_block_size": args.center_block_size,
            "device": args.device,
            "confidence_q25": q25,
            "confidence_q75": q75,
            "confidence_N": n_conf,
        }
        if lda.get("ok", False):
            report.update({
                "w_norm": lda["w_norm"],
                "S_norm": lda["S_norm"],
                "b": lda["b"],
                "threshold_t": lda["threshold_t"],
                "max_w_dot_h": lda["max_w_dot_h"],
                "margin_needed": lda["d_all_w"],
                "d_all_w": lda["d_all_w"],
                "d_all_S": lda["d_all_S"],
                "alpha_all_S": lda["alpha_all_S"],
                "d_mean_w": lda["d_mean_w"],
                "d_mean_S": lda["d_mean_S"],
                "alpha_mean_S": lda["alpha_mean_S"],
                "w_dot_uS": lda["w_dot_uS"],
            })
        if os.path.dirname(args.report_path):
            os.makedirs(os.path.dirname(args.report_path), exist_ok=True)
        with open(args.report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n📝 已写出报告: {args.report_path}")


if __name__ == "__main__":
    main()
