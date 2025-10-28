from typing import Optional, Tuple
import torch
import json


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]
    

@torch.no_grad()
def compute_global_diff_stats(jsonl_path: str, expected_offset: int = 1) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    """
    新增：统计 diff 的 q25 / mean / q75。
    d = ((a - b)^2) / 4.0
    """
    data_json = read_jsonl(jsonl_path)
    diffs = []
    for item in data_json:
        confs = item.get("sentence_confidences") or []
        if not confs or len(confs) <= expected_offset:
            continue
        for j in range(expected_offset, len(confs)):
            try:
                a = float(confs[j])
                b = float(confs[j-1])
            except Exception:
                continue
            d = ((a - b) ** 2) / 4.0
            if torch.isfinite(torch.tensor(d)):
                diffs.append(d)
    if not diffs:
        return None, None, None, 0
    t = torch.tensor(diffs, dtype=torch.float64)
    q25 = t.quantile(0.25).item()
    mean = t.mean().item()
    q75 = t.quantile(0.75).item()
    return q25, mean, q75, len(diffs)


if __name__ == "__main__":
    jsonl_path = "/home/ma-user/work/dataset/outputs_yulin_gy/beta/openPangu-Embedded-7B-V1.1/Math_Math/origin_temp0.7_maxlen16000.jsonl"
    expected_offset = 1
    q25, mean, q75, compute_num = compute_global_diff_stats(jsonl_path)
    print("q25: ", q25)
    print("mean: ", mean)
    print("q75: ", q75)
    print("compute_num: ", compute_num)
