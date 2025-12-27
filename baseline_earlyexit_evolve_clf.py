import os
import json
import argparse
import math
import gc
import glob
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from re import split as rsplit
import random
import numpy as np
import joblib

# ---- silence all warnings/logging (keep your original choices) ----
import os, warnings
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
try:
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
except Exception:
    pass
try:
    import datasets
    datasets.utils.logging.set_verbosity_error()
except Exception:
    pass
try:
    import numpy as np
    np.seterr(all="ignore")
except Exception:
    np = None
# ---- end silence ----

# ------------------------
# Utils
# ------------------------
def parse_optional_float(value):
    """Parse None or float from command line."""
    if value is None:
        return None
    if isinstance(value, str) and value.lower() == 'none':
        return None
    return float(value)

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def write_jsonl_atomic(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp = file_path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    os.replace(tmp, file_path)

def set_seed(seed=42):
    random.seed(seed)
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_npu():
    return hasattr(torch, "npu") and torch.npu.is_available()

def empty_device_cache():
    try:
        if is_npu():
            torch.npu.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

def build_output_paths(args):
    model_basename = os.path.basename(os.path.normpath(args.model_name_or_path))
    output_dir = os.path.join(args.output_path, model_basename, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # Base components
    components = []
    
    # Always include max_generated_tokens
    components.append(f"maxlen{args.max_generated_tokens}")

    dynamic_enabled = (
        args.dynamic_budget_n is not None
        and args.dynamic_budget_n > 0
    )

    # Optional token_budget (early-exit)
    if args.token_budget is not None and not dynamic_enabled:
        components.append(f"tbudget{args.token_budget}")

    # Optional dynamic budget params
    if args.dynamic_budget_n is not None and dynamic_enabled:
        components.append(f"dynN{args.dynamic_budget_n}")
    if args.dynamic_budget_m is not None and dynamic_enabled:
        components.append(f"dynM{args.dynamic_budget_m}")
    
    # Always include seed
    components.append(f"seed{args.seed}")
    
    base_name = "_".join(components)
    return output_dir, base_name

def scan_existing_outputs(output_dir, base_name):
    """
    扫描总表 + 各分片，返回：
      existing_idx(set), existing_q(set)
    用于断点恢复/跳过重复。
    """
    combined = os.path.join(output_dir, f"{base_name}.jsonl")
    shard_glob = os.path.join(output_dir, f"{base_name}.shard*.jsonl")
    files = []
    if os.path.exists(combined):
        files.append(combined)
    files += sorted(glob.glob(shard_glob))

    idx_set, q_set = set(), set()
    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                i = obj.get("idx", None)
                q = obj.get("question", None)
                if isinstance(i, int):
                    idx_set.add(i)
                if isinstance(q, str):
                    q_set.add(q)
    return idx_set, q_set

def merge_all_shards(output_dir, base_name, remove_shards=True):
    """rank 0：合并 *.shard*.jsonl (+ 旧总表)，按 idx 去重，写 {base}.jsonl"""
    shard_files = sorted(glob.glob(os.path.join(output_dir, f"{base_name}.shard*.jsonl")))
    combined_file = os.path.join(output_dir, f"{base_name}.jsonl")

    sources = []
    if os.path.exists(combined_file):
        sources.append(combined_file)
    sources += shard_files

    merged = {}
    for fp in sources:
        with open(fp, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                idx = obj.get("idx", None)
                if isinstance(idx, int) and idx not in merged:
                    merged[idx] = obj

    final = [merged[k] for k in sorted(merged.keys())]
    write_jsonl_atomic(final, combined_file)

    # 删除shard文件
    if remove_shards and shard_files:
        for shard_file in shard_files:
            try:
                os.remove(shard_file)
                print(f"[rank 0] 🗑️  Removed shard: {shard_file}")
            except Exception as e:
                print(f"[WARN][rank 0] Failed to remove {shard_file}: {e}")
    
    return combined_file, len(final)

# ------------------------
# Distributed helpers
# ------------------------

def _auto_backend():
    if is_npu():
        return "hccl"
    if torch.cuda.is_available():
        return "nccl"
    return "gloo"

def init_distributed_if_needed():
    """
    使用 torchrun 时通过 env:// 初始化；单进程则回退为非分布式。
    返回: (is_dist, rank, world_size, local_rank, device)
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_dist = world_size > 1
    rank = 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if is_dist and not dist.is_initialized():
        dist.init_process_group(backend=_auto_backend(), init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    # 设备选择
    device = torch.device("cpu")
    if is_npu():
        torch.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    return is_dist, rank, world_size, local_rank, device

# ------------------------
# Sampling
# ------------------------

@torch.no_grad()
def top_p_sampling_step(last_logits, temperature: float, top_p: float, output_logprobs: bool):
    """
    last_logits: [1, vocab_size] on device
    returns: next_token_id [1,1], next_token_logprob (float)
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0 for sampling.")

    logits = last_logits / temperature
    probs = torch.softmax(logits, dim=-1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    cutoff = (cumsum > top_p)
    cutoff[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
    sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-12)

    next_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_indices.gather(-1, next_sorted_idx)

    if output_logprobs:
        chosen_prob = sorted_probs.gather(-1, next_sorted_idx)
        next_logprob = torch.log(chosen_prob + 1e-12).item()
    else:
        next_logprob = None

    return next_token, next_logprob

def load_model_and_tokenizer(args, device):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True
    )

    dtype = torch.float16 if is_npu() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True
    ).to(device)

    model.eval()
    return model, tokenizer, dtype

def build_stop_ids_contains(tok, needle: str = "\n\n"):
    vocab = tok.get_vocab()
    stop_ids = {int(i) for s, i in vocab.items() if needle in s}
    if not stop_ids:
        raise RuntimeError(f'No tokens containing "{needle}" found in tokenizer vocab.')
    return stop_ids

def safe_predict_proba_1(clf, x_1d: np.ndarray) -> float:
    X = x_1d.reshape(1, -1)
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)
        return float(p[0, 1])
    if hasattr(clf, "decision_function"):
        s = clf.decision_function(X)
        s = float(np.asarray(s).reshape(-1)[0])
        return float(1.0 / (1.0 + np.exp(-s)))
    y = clf.predict(X)
    return float(np.asarray(y).reshape(-1)[0])

def _hidden_state_to_np(hidden_states, layer: int):
    if not hidden_states:
        return None
    L = int(layer) if layer is not None else -1
    if L < 0:
        L = len(hidden_states) - 1
    if L >= len(hidden_states):
        L = len(hidden_states) - 1
    h = hidden_states[L]
    v = h[0, -1, :].detach().to("cpu", dtype=torch.float32)
    return v.numpy()

@torch.no_grad()
def sample_with_tracking(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int,
    dtype: torch.dtype,
    device: torch.device,
    think_budget: int = None,
    answer_budget: int = None,
    think_end_ids: list = None,
    output_logprobs: bool = False,
    stop_ids: set = None,
    clf=None,
    clf_layer: int = None,
    avg_prob: float = None,
    budget_base: float = None,
    answer_budget_ratio: float = None
):
    """
    返回：
      generated_ids: 仅新增生成部分的 token ids (Tensor[T], device)
      token_logprobs: list[float]（若不需要可忽略）
      first_stop_prob: 第一个包含\n\n token 的clf概率（可能为None）
      dynamic_budget: 基于clf概率计算得到的budget（可能为None）
    """
    token_logprobs = []
    generated = []
    past_key_values = None
    first_stop_prob = None
    dynamic_budget = None

    # 先基于prompt建立KV
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, past_key_values=None)
        past_key_values = outputs.past_key_values
        # avoid keeping full prompt logits alive
        last_logits = outputs.logits[:, -1, :].contiguous()
    del outputs

    total_generated = 0
    answer_generated = 0
    in_answer_phase = False
    think_end_inserted = False

    def _maybe_set_dynamic_budget(prob: float):
        nonlocal dynamic_budget, think_budget, answer_budget
        if prob is None or avg_prob is None or budget_base is None:
            return
        if avg_prob <= 0:
            return
        ratio = prob / avg_prob
        if not math.isfinite(ratio):
            return
        raw = budget_base * ratio
        budget = int(round(raw))
        if budget < 1:
            budget = 1
        if budget > max_new_tokens:
            budget = max_new_tokens
        dynamic_budget = budget
        think_budget = budget
        if (
            answer_budget is None
            and answer_budget_ratio is not None
            and answer_budget_ratio > 0
        ):
            answer_budget = max(1, int(round(budget * answer_budget_ratio)))

    def _append_forced_token(token_id: int):
        nonlocal past_key_values, last_logits, total_generated, answer_generated
        token_tensor = torch.tensor([[token_id]], device=device)
        if output_logprobs:
            logprob = torch.log_softmax(last_logits, dim=-1)[0, token_id].item()
        else:
            logprob = None
        token_logprobs.append(logprob)
        generated.append(token_id)
        total_generated += 1
        if in_answer_phase:
            answer_generated += 1
        with torch.inference_mode():
            out = model(
                input_ids=token_tensor,
                attention_mask=None,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = out.past_key_values
            last_logits = out.logits[:, -1, :].contiguous()
        del out

    while total_generated < max_new_tokens:
        if (not in_answer_phase) and (think_budget is not None) and total_generated >= think_budget:
            if think_end_ids and not think_end_inserted:
                for token_id in think_end_ids:
                    if total_generated >= max_new_tokens:
                        break
                    _append_forced_token(token_id)
                think_end_inserted = True
            in_answer_phase = True

        if in_answer_phase and answer_budget is not None and answer_generated >= answer_budget:
            break

        # 采样一步
        next_token, next_logprob = top_p_sampling_step(
            last_logits,
            temperature,
            top_p,
            output_logprobs=output_logprobs
        )
        token_logprobs.append(next_logprob)

        last_id = next_token.item()
        generated.append(last_id)
        total_generated += 1
        if in_answer_phase:
            answer_generated += 1

        need_clf = (
            first_stop_prob is None
            and stop_ids is not None
            and clf is not None
            and last_id in stop_ids
        )
        if eos_token_id is not None and last_id == eos_token_id and not need_clf:
            break

        # 基于上一步 token 继续前推（复用KV）
        with torch.inference_mode():
            out = model(
                input_ids=next_token,  # [1,1]
                attention_mask=None,
                use_cache=True,
                past_key_values=past_key_values,
                output_hidden_states=need_clf,
            )
            past_key_values = out.past_key_values
            last_logits = out.logits[:, -1, :].contiguous()

        if need_clf:
            rep = _hidden_state_to_np(out.hidden_states, clf_layer)
            if rep is not None:
                first_stop_prob = safe_predict_proba_1(clf, rep)
                _maybe_set_dynamic_budget(first_stop_prob)
        del out

        if eos_token_id is not None and last_id == eos_token_id:
            break

    if len(generated) == 0:
        return torch.empty(0, dtype=torch.long, device=device), [], None, None

    del past_key_values, last_logits
    if "out" in locals():
        del out
    if "next_token" in locals():
        del next_token
    empty_device_cache()

    return torch.tensor(generated, dtype=torch.long, device=device), token_logprobs, first_stop_prob, dynamic_budget

# ------------------------
# Core worker
# ------------------------

def worker(args, rank, world_size, local_rank, device):
    # Dataset
    dataset_path = os.path.join(args.dataset_dir, args.dataset, 'test.jsonl')
    questions = read_jsonl(dataset_path)
    N = len(questions)

    # Output paths
    output_dir, base_name = build_output_paths(args)
    shard_file = os.path.join(output_dir, f"{base_name}.shard{rank:03d}.jsonl")

    # 扫描现有结果（总表 + 全部 shard），用于断点恢复/跳过
    existing_idx_global, existing_q_global = scan_existing_outputs(output_dir, base_name)

    # Model & tokenizer
    model, tokenizer, dtype = load_model_and_tokenizer(args, device)

    dynamic_enabled = (
        args.dynamic_budget_n is not None
        and args.dynamic_budget_n > 0
    )
    if dynamic_enabled and args.dynamic_budget_m is None:
        raise ValueError("--dynamic_budget_m is required when dynamic_budget_n>0.")
    if dynamic_enabled and args.token_budget is not None and rank == 0:
        print("[rank 0] [INFO] dynamic budget enabled; ignoring --token_budget during sampling.")

    insert_clf = None
    clf_layer = None
    stop_ids = None
    if dynamic_enabled:
        if not args.clf or not args.meta:
            raise ValueError("--clf and --meta are required when dynamic_budget_n>0.")
        insert_clf = joblib.load(args.clf)
        with open(args.meta, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta_best_layer = int(meta.get("best_layer", -1))
        clf_layer = int(args.clf_layer) if args.clf_layer is not None else meta_best_layer
        stop_ids = build_stop_ids_contains(tokenizer, needle="\n\n")

    avg_len = None
    avg_prob = None
    budget_base = None
    bootstrap_sum = 0
    bootstrap_count = 0
    bootstrap_prob_sum = 0.0
    bootstrap_prob_count = 0
    bootstrap_n = 0

    def _compute_budget_base(avg_len_val: float) -> float:
        return avg_len_val * (args.dynamic_budget_m / 100.0)

    think_budget = None
    answer_budget = None
    think_end_ids = None
    answer_budget_ratio = None
    if args.token_budget is not None and args.token_budget > 0 and not dynamic_enabled:
        think_budget = int(args.token_budget)
        # answer_budget = max(1, think_budget // 4)  # optional
    if (args.token_budget is not None and args.token_budget > 0) or dynamic_enabled:
        think_end_ids = tokenizer.encode("\n[unused17]\n\n", add_special_tokens=False)
    if dynamic_enabled:
        answer_budget_ratio = args.answer_budget_ratio

    def _run_generation(
        i,
        current_think_budget,
        avg_prob_val=None,
        budget_base_val=None,
        stop_ids_val=None,
        clf_val=None,
        clf_layer_val=None
    ):
        q = questions[i]
        qtext = q.get("problem", "")

        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": qtext}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        try:
            with torch.inference_mode():
                gen_ids, _step_logprobs, first_stop_prob, used_budget = sample_with_tracking(
                    model=model,
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])),
                    max_new_tokens=args.max_generated_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_token_id=tokenizer.eos_token_id,
                    dtype=dtype,
                    device=device,
                    think_budget=current_think_budget,
                    answer_budget=answer_budget,
                    think_end_ids=think_end_ids,
                    output_logprobs=False,
                    stop_ids=stop_ids_val,
                    clf=clf_val,
                    clf_layer=clf_layer_val,
                    avg_prob=avg_prob_val,
                    budget_base=budget_base_val,
                    answer_budget_ratio=answer_budget_ratio
                )
        except torch.npu.OutOfMemoryError:
            print(f"[OOM][rank {rank}] Skipping idx={i} : {qtext}...")
            empty_device_cache()
            return None

        full_ids = torch.cat([inputs["input_ids"][0], gen_ids], dim=0)
        response_text = tokenizer.decode(
            full_ids[inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        gen_len = int(gen_ids.numel())

        del inputs, gen_ids
        empty_device_cache()
        return q, qtext, response_text, gen_len, first_stop_prob, used_budget

    if dynamic_enabled:
        bootstrap_n = min(args.dynamic_budget_n, N)
        if rank == 0 and bootstrap_n > 0:
            bootstrap_pbar = tqdm(
                total=bootstrap_n,
                desc=f"Rank {rank} Bootstrap",
                position=rank,
                leave=True
            )
            for i in range(bootstrap_n):
                out = _run_generation(
                    i,
                    think_budget,
                    stop_ids_val=stop_ids,
                    clf_val=insert_clf,
                    clf_layer_val=clf_layer
                )
                bootstrap_pbar.update(1)
                if out is None:
                    continue
                q, qtext, response_text, gen_len, first_stop_prob, _used_budget = out

                bootstrap_sum += gen_len
                bootstrap_count += 1
                if first_stop_prob is not None:
                    bootstrap_prob_sum += first_stop_prob
                    bootstrap_prob_count += 1

                if (i in existing_idx_global) or (qtext in existing_q_global):
                    continue

                result = {
                    "idx": i,
                    "question": qtext,
                    "generated_responses": [response_text],
                    "gold_answer": q.get("answer", ""),
                    "metadata": {
                        "think_budget": think_budget,
                        "used_dynamic_budget": None,
                        "first_stop_prob": first_stop_prob,
                    }
                }

                with open(shard_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            bootstrap_pbar.close()

            if bootstrap_count > 0:
                avg_len = bootstrap_sum / bootstrap_count
                budget_base = _compute_budget_base(avg_len)
                print(
                    f"[rank 0] budget base set to {budget_base:.2f} "
                    f"(avg_len={avg_len:.2f}, m={args.dynamic_budget_m}%)"
                )
            else:
                print("[rank 0] [WARN] budget base not set (bootstrap_count=0).")

            if bootstrap_prob_count > 0:
                avg_prob = bootstrap_prob_sum / bootstrap_prob_count
                print(
                    f"[rank 0] avg first-stop clf prob set to {avg_prob:.6f} "
                    f"(count={bootstrap_prob_count})"
                )
            else:
                print("[rank 0] [WARN] avg first-stop clf prob not set (bootstrap_prob_count=0).")

        if dist.is_initialized():
            avg_len_value = avg_len if avg_len is not None else -1.0
            avg_prob_value = avg_prob if avg_prob is not None else -1.0
            avg_tensor = torch.tensor([avg_len_value, avg_prob_value], device=device, dtype=torch.float32)
            dist.broadcast(avg_tensor, src=0)
            if rank != 0:
                avg_len = float(avg_tensor[0].item())
                avg_prob = float(avg_tensor[1].item())
                if avg_len < 0:
                    avg_len = None
                if avg_prob < 0:
                    avg_prob = None
            if avg_len is not None:
                budget_base = _compute_budget_base(avg_len)

    if dynamic_enabled:
        remaining_indices = list(range(bootstrap_n, N))
    else:
        remaining_indices = list(range(N))

    # 自动划分样本（不再手写 i % world_size）
    if remaining_indices:
        sampler = DistributedSampler(
            list(range(len(remaining_indices))),
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False
        )
        sampler.set_epoch(0)
        raw_idx = list(iter(sampler))
        # 去重以防 padding
        seen, my_indices = set(), []
        for pos in raw_idx:
            if pos < len(remaining_indices) and pos not in seen:
                seen.add(pos)
                my_indices.append(remaining_indices[pos])
    else:
        my_indices = []

    if rank == 0:
        print(f"[rank {rank}] world_size = {world_size}")
        print(f"[rank {rank}] output_dir = {output_dir}")

    print(f"[rank {rank}] shard_file = {shard_file}")
    print(f"[rank {rank}] loaded existing: idx={len(existing_idx_global)}, question={len(existing_q_global)}")
    print(f"[rank {rank}] will process {len(my_indices)} items")

    pbar = tqdm(total=len(my_indices), desc=f"Rank {rank} DP Inference", position=rank, leave=True)

    for i in my_indices:
        q = questions[i]
        qtext = q.get("problem", "")

        # 断点恢复：若已存在则跳过
        if (i in existing_idx_global) or (qtext in existing_q_global):
            pbar.update(1)
            continue

        current_think_budget = None if dynamic_enabled else think_budget

        out = _run_generation(
            i,
            current_think_budget,
            avg_prob_val=avg_prob,
            budget_base_val=budget_base,
            stop_ids_val=stop_ids,
            clf_val=insert_clf,
            clf_layer_val=clf_layer
        )
        if out is None:
            pbar.update(1)
            continue
        q, qtext, response_text, _gen_len, first_stop_prob, used_budget = out

        used_budget_final = used_budget if used_budget is not None else current_think_budget
        used_dynamic_budget = used_budget_final if dynamic_enabled else None

        result = {
            "idx": i,
            "question": qtext,
            "generated_responses": [response_text],
            "gold_answer": q.get("answer", ""),
            "metadata": {
                "think_budget": used_budget_final,
                "used_dynamic_budget": used_dynamic_budget,
                "first_stop_prob": first_stop_prob,
            }
        }

        with open(shard_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

        pbar.update(1)

    pbar.close()
    print(f"[rank {rank}] ✅ Done. Shard saved to {shard_file}")

# ------------------------
# Evaluation
# ------------------------
def evaluate_and_save(args, combined_file):
    from utils.data_loader import load_data
    from utils.parser import parse_ground_truth, extract_answer
    from utils.grader import check_is_correct
    from math import comb

    # --------- helpers ---------
    def _extract_first_text(gen):
        """尽量兼容多种结构，取首个文本用于长度统计；评测正确性仍看所有候选。"""
        if isinstance(gen, str):
            return gen
        if isinstance(gen, dict):
            for key in ("text", "content", "generated_response", "generated_text", "output", "message", "response"):
                v = gen.get(key)
                if isinstance(v, str):
                    return v
        if isinstance(gen, list) and gen:
            for item in gen:
                t = _extract_first_text(item)
                if t:
                    return t
        return ""

    def _extract_all_texts(gens):
        out = []
        for g in gens:
            if isinstance(g, str):
                out.append(g)
            elif isinstance(g, dict):
                for key in ("text", "content", "generated_response", "generated_text", "output", "message", "response"):
                    v = g.get(key)
                    if isinstance(v, str):
                        out.append(v); break
            elif isinstance(g, list) and g:
                # 取子项中的首个字符串
                t = _extract_first_text(g)
                if t:
                    out.append(t)
        return out
    
    # --------- load ---------
    outputs = read_jsonl(combined_file)
    outputs_by_idx = {o.get("idx", i): o for i, o in enumerate(outputs)}
    examples = load_data(args.dataset, args.split, args.dataset_dir)

    # --------- correctness & pass@k ---------
    total = len(examples)
    correct_cnt = 0
    pass_at_k_vals = []
    wrong_ids = []
    for i in tqdm(range(total), desc="Evaluating", leave=False):
        d = examples[i]
        gt_cot, gt_ans = parse_ground_truth(d, args.dataset)
        out = outputs_by_idx.get(i)
        if not out:
            wrong_ids.append(d.get("id", i))
            continue
        texts = _extract_all_texts(out.get("generated_responses", []))
        if not texts:
            wrong_ids.append(d.get("id", i))
            continue
        gen_answers = [extract_answer(t, args.dataset) for t in texts]
        is_correct_list = [check_is_correct(a, gt_ans) for a in gen_answers]
        if any(is_correct_list):
            correct_cnt += 1
        else:
            wrong_ids.append(d.get("id", i))
        if len(is_correct_list) > 1:
            c = sum(is_correct_list)
            n = len(is_correct_list)
            if c > 0:
                if n - c < args.k:
                    val = 1.0
                else:
                    val = 1.0 - (comb(n - c, args.k) / comb(n, args.k))
                pass_at_k_vals.append(val)
            else:
                pass_at_k_vals.append(0.0)

    acc = correct_cnt / total if total else 0.0
    metrics = {
        "dataset": args.dataset,
        "split": args.split,
        "total": total,
        "generated": len(outputs),
        "correct": correct_cnt,
        "accuracy": acc,
        "k": args.k
    }
    if pass_at_k_vals:
        metrics[f"pass@{args.k}"] = sum(pass_at_k_vals) / len(pass_at_k_vals)
    else:
        metrics[f"pass@{args.k}"] = acc  # 单样本时退化为 Acc

    # 添加完整性检查信息
    is_complete = len(outputs) == total
    metrics["is_complete"] = is_complete
    if not is_complete:
        metrics["missing_count"] = total - len(outputs)
    
    # --------- token length stats (按原始逻辑) ---------
    # 统计基于 outputs（与原脚本一致）
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=True
    )

    test_num = len(outputs)
    resp_word_counts = []
    full_token_counts = []
    think_token_counts = []
    think_found = 0
    fallback_full = 0

    for data in outputs:
        gens = data.get("generated_responses", [])
        text = _extract_first_text(gens) if gens else ""
        # 1) 词数
        resp_word_counts.append(len(text.split()) if text else 0)
        # 2) 全文 token 数（与原始逻辑一致：不加 special tokens）
        if text:
            full_tokens_len = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        else:
            full_tokens_len = 0
        full_token_counts.append(full_tokens_len)
        # 3) think 段 token 数：以 [unused17] 截断，找不到则用全文
        lower = text.lower() if text else ""
        idx = lower.find("[unused17]")  # Pangu 的思维段落边界
        if idx != -1:
            think_text = text[:idx]
            think_found += 1
        else:
            think_text = text
            if text:
                fallback_full += 1
        if think_text:
            think_tokens_len = len(tokenizer(think_text, add_special_tokens=False)["input_ids"])
        else:
            think_tokens_len = 0
        think_token_counts.append(think_tokens_len)

    avg_resp_words = (sum(resp_word_counts) / test_num) if test_num else 0.0
    avg_full_tokens = (sum(full_token_counts) / test_num) if test_num else 0.0
    avg_think_tokens = (sum(think_token_counts) / test_num) if test_num else 0.0

    metrics["token_stats"] = {
        "samples": test_num,
        "avg_word_count": avg_resp_words,
        "avg_full_token_count": avg_full_tokens,
        "avg_think_token_count": avg_think_tokens,
        "think_found": think_found,
        "think_fallback_full": fallback_full
    }

    # --------- save ---------
    output_dir, base_name = build_output_paths(args)
    metrics_path = os.path.join(output_dir, f"{base_name}.metrics.json")
    wrong_ids_path = os.path.join(output_dir, f"{base_name}.wrong_ids.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(wrong_ids_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(wrong_ids), "ids": wrong_ids}, f, ensure_ascii=False, indent=2)
    print(f"[rank 0] ✅ Metrics saved to: {metrics_path}")
    print(f"[rank 0] ✅ Wrong IDs saved to: {wrong_ids_path} (count={len(wrong_ids)})")
    return metrics_path, wrong_ids_path

# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_generated_tokens', type=int, default=16000)
    parser.add_argument('--token_budget', type=int, default=None)
    parser.add_argument('--dynamic_budget_n', type=int, default=None)
    parser.add_argument('--dynamic_budget_m', type=float, default=None)
    parser.add_argument("--clf", type=str, default="", help="Path to clf.joblib for first \\n\\n prob")
    parser.add_argument("--meta", type=str, default="", help="Path to clf meta json")
    parser.add_argument("--clf_layer", type=int, default=None, help="Override meta.best_layer")
    parser.add_argument(
        "--answer_budget_ratio",
        type=float,
        default=0.25,
        help="When dynamic budget is enabled, cap answer tokens to ratio * think_budget (0 disables)."
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--k", type=int, default=1, help="Value of k for pass@k calculation")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()
    
    set_seed(args.seed)

    is_dist, rank, world_size, local_rank, device = init_distributed_if_needed()

    # 打印接收到的参数
    if rank == 0:
        print("=" * 80)
        print("Received Arguments:")
        print("=" * 80)
        for arg, value in vars(args).items():
            print(f"  {arg} = {value}")
        print("=" * 80)

    worker(args, rank, world_size, local_rank, device)

    # 同步 & 汇总
    if is_dist:
        dist.barrier()
    if rank == 0:
        output_dir, base_name = build_output_paths(args)
        combined_file, count = merge_all_shards(output_dir, base_name, remove_shards=True)
        print(f"[rank 0] ✅ Merged {count} entries to: {combined_file}")
        evaluate_and_save(args, combined_file)
    if is_dist and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
