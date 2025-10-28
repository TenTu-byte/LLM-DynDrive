from typing import Optional, Tuple
import torch
import json


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]
    

@torch.no_grad()
def compute_global_diff_stats(jsonl_path: str) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    """
    新增：统计 diff 的 q25 / mean / q75。
    d = ((a - b)^2) / 4.0
    """
    data_json = read_jsonl(jsonl_path)
    diffs = []
    for item in data_json:
        confs = item.get("sentence_confidences") or []
        # conf_num = item.get("confidence_num")
        # hidden_num = item.get("step_hidden_num")
        # idx = item.get("idx")

        # try:
        #     conf_num_i = int(conf_num) if conf_num is not None else None
        #     hidden_num_i = int(hidden_num) if hidden_num is not None else None
        # except Exception:
        #     print(f"[ERROR] idx={idx}: invalid conf_num/hidden_num (conf_num={conf_num}, hidden_num={hidden_num}); skip")
        #     continue

        # if conf_num_i is None or hidden_num_i is None or conf_num_i != hidden_num_i + expected_offset:
        #     print(f"[ERROR] idx={idx}: conf_num ({conf_num_i}) != hidden_num ({hidden_num_i}) + expected_offset ({expected_offset}); skip")
        #     continue

        if not confs or len(confs) <= 1:
            continue
        for j in range(1, len(confs)): 
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
    jsonl_path = "/home/ma-user/work/dataset/outputs_yulin_gy/test3/openPangu-Embedded-7B-V1.1/Math_Math/origin_temp0.7_maxlen16000.jsonl"
    expected_offset = 0
    q25, mean, q75, compute_num = compute_global_diff_stats(jsonl_path)
    print("q25: ", q25)
    print("mean: ", mean)
    print("q75: ", q75)
    print("compute_num: ", compute_num)
