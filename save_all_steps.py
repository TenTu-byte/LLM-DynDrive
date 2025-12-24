# -*- coding: utf-8 -*-

import os
import json
import argparse
import gc
import glob
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ------------------------
# Small utils
# ------------------------

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_npu():
    return hasattr(torch, "npu") and torch.npu.is_available()


def _clear_device():
    if is_npu():
        try:
            torch.npu.synchronize()
        except Exception:
            pass
        try:
            torch.npu.empty_cache()
        except Exception:
            pass
        try:
            torch.npu.ipc_collect()
        except Exception:
            pass
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            out.append(json.loads(s))
    return out


def write_jsonl_atomic(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp = file_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(tmp, file_path)


def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _oom_exception_types():
    errors = [torch.cuda.OutOfMemoryError]
    if hasattr(torch, "npu") and hasattr(torch.npu, "OutOfMemoryError"):
        errors.append(torch.npu.OutOfMemoryError)
    return tuple(errors)


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
    """使用 torchrun 跨节点启动时通过 env:// 初始化；否则单进程回退。"""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_dist = world_size > 1
    rank = 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if is_dist and not dist.is_initialized():
        dist.init_process_group(backend=_auto_backend(), init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    device = torch.device("cpu")
    if is_npu():
        torch.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    return is_dist, rank, world_size, local_rank, device


# ------------------------
# Shard helpers
# ------------------------

def build_output_paths(args):
    model_basename = os.path.basename(os.path.normpath(args.model_name_or_path))
    output_dir = os.path.join(args.output_path, model_basename, args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    components = []
    if args.run_id:
        components.append(args.run_id.strip())
    components.append("results")
    base_name = "_".join(components)
    return output_dir, base_name


def scan_existing_outputs(output_dir, base_name):
    """
    扫描总表 + 各分片, 汇总已有的 idx 与 question, 构建 idx->obj / q->obj。
    """
    combined = os.path.join(output_dir, f"{base_name}.jsonl")
    shard_glob = os.path.join(output_dir, f"{base_name}.shard*.jsonl")

    files = []
    if os.path.exists(combined):
        files.append(combined)
    files += sorted(glob.glob(shard_glob))

    idx_set, q_set = set(), set()
    idx_map, q_map = {}, {}
    total = 0

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                total += 1
                try:
                    obj = json.loads(s)
                except Exception:
                    continue
                i = obj.get("idx", None)
                q = obj.get("question", None)
                if isinstance(i, int):
                    if i not in idx_map:
                        idx_map[i] = obj
                    idx_set.add(i)
                if isinstance(q, str):
                    if q not in q_map:
                        q_map[q] = obj
                    q_set.add(q)
    return idx_set, q_set, idx_map, q_map, total


def _reconstruct_full_text(tokenizer, sys_prompt, question_text, response_text):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + (response_text or "")


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
        with open(fp, "r", encoding="utf-8") as f:
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

    if remove_shards and shard_files:
        for shard_file in shard_files:
            try:
                os.remove(shard_file)
                print(f"[rank 0] 🗑️  Removed shard: {shard_file}")
            except Exception as e:
                print(f"[WARN][rank 0] Failed to remove {shard_file}: {e}")

    return combined_file, len(final)


# ------------------------
# Core helpers
# ------------------------

def _find_all_subseq_positions(seq, subseq):
    """
    Return all start indices where subseq matches seq (allow overlaps).
    seq, subseq: list[int]
    """
    if not subseq:
        return []
    L, M = len(seq), len(subseq)
    out = []
    for s in range(0, L - M + 1):
        if seq[s:s + M] == subseq:
            out.append(s)
    return out


def build_stop_ids_contains(tok, needle: str = "\n\n"):
    """
    Build a set of token ids whose token string contains `needle`.
    For Qwen2/Qwen3 TokenizerFast:
      "\\n\\n"  -> 'ĊĊ'
      " \\n\\n" -> 'ĠĊĊ'
    Both contain 'ĊĊ'.
    """
    vocab = tok.get_vocab()  # token_str -> id
    stop_ids = {int(i) for s, i in vocab.items() if needle in s}
    if not stop_ids:
        raise RuntimeError(f'No tokens containing "{needle}" found in tokenizer vocab.')
    return stop_ids


def first_index_in_set(seq, id_set):
    """
    Return first index t where seq[t] in id_set, else None.
    seq: List[int]
    """
    s = set(int(x) for x in id_set)
    for t, tid in enumerate(seq):
        if int(tid) in s:
            return t
    return None


def _encode_no_special(tok, text: str):
    try:
        ids = tok.encode(text, add_special_tokens=False)
        return [int(x) for x in ids]
    except Exception:
        return []


def _find_earliest_think_end(gen_ids_list, tokenizer):
    """
    Find earliest start index of a '[unused17]'-like marker in generated token ids.
    Return (think_end_gen, matched_pattern_ids). If not found, return (len(gen), []).
    """
    patterns = [
        "[unused17]",
        " [unused17]",
        "\n[unused17]",
        "\n\n[unused17]",
    ]
    best = None
    best_pat = []
    for p in patterns:
        pat_ids = _encode_no_special(tokenizer, p)
        if not pat_ids:
            continue
        pos = _find_all_subseq_positions(gen_ids_list, pat_ids)
        if not pos:
            continue
        start = int(pos[0])
        if best is None or start < best:
            best = start
            best_pat = pat_ids
    if best is None:
        return len(gen_ids_list), []
    return best, best_pat


# ------------------------
# Old function (kept for compatibility/reference)
# ------------------------

def save_double_newline_token_hs_first_only(
    model,
    tokenizer,
    full_ids_1xT: torch.Tensor,   # [1, T] on device
    prompt_len: int,
    save_path: str,
    hs_device: str = "auto",
):
    """
    Save all-layer hidden states at ONLY the FIRST occurrence of a token whose
    token string contains "ĊĊ" (e.g., 'ĊĊ' or 'ĠĊĊ') in generated tokens.
    """
    import os as _os

    stop_ids = build_stop_ids_contains(tokenizer, needle="ĊĊ")

    gen_ids_list = full_ids_1xT[0, prompt_len:].detach().cpu().tolist()
    t0 = first_index_in_set(gen_ids_list, stop_ids)

    pos_full = []
    hit_id = None
    if t0 is not None:
        hit_id = int(gen_ids_list[t0])
        pos_full = [prompt_len + t0]

    hs_out = {}

    def _forward_on(device):
        with torch.inference_mode():
            out = model(full_ids_1xT.to(device), output_hidden_states=True, use_cache=False, return_dict=True)
        if len(pos_full) == 0:
            return
        pos = torch.tensor(pos_full, dtype=torch.long, device=out.hidden_states[0].device)
        for lid, h in enumerate(out.hidden_states):  # each: [1, T, H]
            hs_out[int(lid)] = h[0].index_select(0, pos).to("cpu", non_blocking=True)  # [1, H]

    orig_device = next(model.parameters()).device
    try:
        if hs_device in ("auto", "npu") and is_npu():
            _forward_on(orig_device)
        elif hs_device in ("auto", "cuda") and torch.cuda.is_available():
            _forward_on(orig_device)
        else:
            if orig_device.type != "cpu":
                model.to("cpu")
            _forward_on(torch.device("cpu"))
    except _oom_exception_types():
        if hs_device in ("cuda", "npu"):
            raise
        _clear_device()
        model.to("cpu")
        _forward_on(torch.device("cpu"))
    finally:
        if next(model.parameters()).device != orig_device:
            model.to(orig_device)

    _os.makedirs(_os.path.dirname(save_path), exist_ok=True)
    torch.save({"pos_full": pos_full, "pattern_ids": ([] if hit_id is None else [hit_id]), "hs": hs_out}, save_path)


# ------------------------
# NEW: save ALL "\n\n" token hidden states BEFORE [unused17]
# ------------------------

def save_double_newline_token_hs_before_think_all(
    model,
    tokenizer,
    full_ids_cpu_1xT: torch.Tensor,   # [1, T] on CPU (recommended)
    prompt_len: int,
    save_path: str,
    hs_device: str = "auto",
    block_size: int = 256,
):
    r"""
    Save all-layer hidden states at ALL occurrences of a token whose token string contains "\n\n" BUT ONLY those occurrences strictly BEFORE the first '[unused17]' marker.

    Also save remain_length list aligned 1-to-1 with every '\n\n' hit:
      remain_length[k] = (#tokens remaining in the thinking part, i.e., until [unused17]) after that '\n\n' token.

    File content:
      {
        "pos_gen": [int]*K,          # positions in generated tokens (0-based)
        "pos_full": [int]*K,         # positions in full sequence (prompt+gen)
        "pattern_ids": [int]*K,      # the hit token id at each position
        "think_end_gen": int,        # where [unused17] starts in generated tokens (or len(gen) if not found)
        "think_end_full": int,       # prompt_len + think_end_gen
        "think_pattern_ids": [int],  # which pattern ids matched (maybe empty)
        "remain_length": [int]*K,    # aligned with pos_gen / pos_full
        "hs": {layer_id: Tensor[K,H]}
      }
    """
    import os as _os

    stop_ids = build_stop_ids_contains(tokenizer, needle="\n\n")

    # work in python lists on CPU
    full_ids_list = full_ids_cpu_1xT[0].tolist()
    gen_ids_list = full_ids_list[prompt_len:]
    gen_len = len(gen_ids_list)

    think_end_gen, think_pat_ids = _find_earliest_think_end(gen_ids_list, tokenizer)  # start index of [unused17]
    think_end_gen = int(min(max(think_end_gen, 0), gen_len))
    think_end_full = prompt_len + think_end_gen

    # collect all hits before [unused17]
    pos_gen = []
    pos_full = []
    hit_token_ids = []
    remain_length = []
    for t, tid in enumerate(gen_ids_list):
        if t >= think_end_gen:
            break
        if int(tid) in stop_ids:
            pos_gen.append(int(t))
            pos_full.append(int(prompt_len + t))
            hit_token_ids.append(int(tid))
            # remaining tokens in thinking part AFTER this '\n\n' token
            remain_length.append(int(think_end_gen - (t + 1)))

    K = len(pos_full)

    # If no hits, still write a file for consistency
    hs_out = {}
    if K == 0:
        _os.makedirs(_os.path.dirname(save_path), exist_ok=True)
        torch.save(
            {
                "pos_gen": pos_gen,
                "pos_full": pos_full,
                "pattern_ids": hit_token_ids,
                "think_end_gen": think_end_gen,
                "think_end_full": think_end_full,
                "think_pattern_ids": think_pat_ids,
                "remain_length": remain_length,
                "hs": hs_out,
            },
            save_path,
        )
        return

    # Decide compute device
    orig_device = next(model.parameters()).device
    compute_device = torch.device("cpu")
    if hs_device in ("auto", "npu") and is_npu():
        compute_device = orig_device
    elif hs_device in ("auto", "cuda") and torch.cuda.is_available():
        compute_device = orig_device

    # Move model if needed
    moved = False
    if next(model.parameters()).device != compute_device:
        model.to(compute_device)
        moved = True

    # We do cache+chunk forward to avoid building full hidden states for the whole sequence
    # Gather hidden states only at pos_full.
    # Allocate per-layer list[K] then stack -> [K,H]
    per_layer = None  # will become List[List[Tensor]]
    pos_full_sorted = pos_full  # already increasing by construction

    try:
        with torch.inference_mode():
            T = len(full_ids_list)
            past = None
            cur = 0  # pointer in pos_full_sorted

            # to avoid repeated CPU->GPU copies, we slice from CPU tensor and move chunk to device
            for start in range(0, T, max(1, int(block_size))):
                end = min(T, start + max(1, int(block_size)))
                chunk = full_ids_cpu_1xT[:, start:end].to(compute_device, non_blocking=True)

                if past is None:
                    out = model(
                        input_ids=chunk,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                else:
                    out = model(
                        input_ids=chunk,
                        past_key_values=past,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                past = out.past_key_values

                if per_layer is None:
                    # out.hidden_states length = num_layers+1 (emb + each layer)
                    per_layer = [[] for _ in range(len(out.hidden_states))]

                # consume all target positions that fall into [start, end)
                while cur < K and pos_full_sorted[cur] < end:
                    p = pos_full_sorted[cur]
                    if p >= start:
                        local = int(p - start)  # index in this chunk
                        # each hidden: [1, chunk_len, H] -> take [0, local]
                        for lid, h in enumerate(out.hidden_states):
                            per_layer[lid].append(h[0, local].detach().to("cpu", non_blocking=True))
                    cur += 1

                # early stop if already collected all
                if cur >= K:
                    break

        # finalize
        if per_layer is not None:
            for lid, vecs in enumerate(per_layer):
                if len(vecs) != K:
                    # safety: if something went wrong, keep only what we have
                    if len(vecs) == 0:
                        continue
                hs_out[int(lid)] = torch.stack(vecs, dim=0)  # [K,H]

    except TypeError:
        # Fallback (less memory safe): full forward then index_select.
        # Keep it as a compatibility fallback for weird remote-code models.
        with torch.inference_mode():
            out = model(full_ids_cpu_1xT.to(compute_device), output_hidden_states=True, use_cache=False, return_dict=True)
        pos = torch.tensor(pos_full_sorted, dtype=torch.long, device=out.hidden_states[0].device)
        for lid, h in enumerate(out.hidden_states):
            hs_out[int(lid)] = h[0].index_select(0, pos).to("cpu", non_blocking=True)

    except _oom_exception_types():
        if hs_device in ("cuda", "npu"):
            raise
        _clear_device()
        model.to("cpu")
        moved = True
        compute_device = torch.device("cpu")
        # retry on CPU (chunk)
        hs_out = {}
        per_layer = None
        with torch.inference_mode():
            T = len(full_ids_list)
            past = None
            cur = 0
            for start in range(0, T, max(1, int(block_size))):
                end = min(T, start + max(1, int(block_size)))
                chunk = full_ids_cpu_1xT[:, start:end].to(compute_device)

                if past is None:
                    out = model(input_ids=chunk, use_cache=True, output_hidden_states=True, return_dict=True)
                else:
                    out = model(
                        input_ids=chunk,
                        past_key_values=past,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                past = out.past_key_values

                if per_layer is None:
                    per_layer = [[] for _ in range(len(out.hidden_states))]

                while cur < K and pos_full_sorted[cur] < end:
                    p = pos_full_sorted[cur]
                    if p >= start:
                        local = int(p - start)
                        for lid, h in enumerate(out.hidden_states):
                            per_layer[lid].append(h[0, local].detach().to("cpu"))
                    cur += 1
                if cur >= K:
                    break

        if per_layer is not None:
            for lid, vecs in enumerate(per_layer):
                if len(vecs) == K:
                    hs_out[int(lid)] = torch.stack(vecs, dim=0)

    finally:
        # restore model device
        if moved and next(model.parameters()).device != orig_device:
            model.to(orig_device)

    _os.makedirs(_os.path.dirname(save_path), exist_ok=True)
    torch.save(
        {
            "pos_gen": pos_gen,
            "pos_full": pos_full,
            "pattern_ids": hit_token_ids,
            "think_end_gen": think_end_gen,
            "think_end_full": think_end_full,
            "think_pattern_ids": think_pat_ids,
            "remain_length": remain_length,
            "hs": hs_out,
        },
        save_path,
    )


# ------------------------
# Model load
# ------------------------

def load_model_tokenizer(model_name_or_path, trust_remote_code, device):
    tok = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=False,
        trust_remote_code=trust_remote_code,
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype="auto",
        local_files_only=True,
    )
    model.to(device)
    model.eval()
    return model, tok


def print_double_newline_token_info_once(tok, rank: int):
    """
    Print how "\\n\\n" is tokenized: ids + token strings.
    """
    try:
        pattern_ids = tok.encode("\n\n", add_special_tokens=False)
        pattern_ids = [int(x) for x in pattern_ids]
        pattern_toks = tok.convert_ids_to_tokens(pattern_ids)
        print(f"[rank {rank}] tokenizer('\\n\\n') pattern_ids = {pattern_ids}")
        print(f"[rank {rank}] tokenizer('\\n\\n') pattern_tokens = {pattern_toks}")
    except Exception as e:
        print(f"[rank {rank}] failed to print tokenizer('\\n\\n') info: {e}")


# ------------------------
# Worker
# ------------------------

def worker(args, rank, world_size, device):
    set_seeds(args.seed + rank)

    dataset_path = os.path.join(args.dataset_dir, args.dataset, "train.jsonl")
    data = read_jsonl(dataset_path)[:500]
    N = len(data)

    output_dir, base_name = build_output_paths(args)
    shard_path = os.path.join(output_dir, f"{base_name}.shard{rank:03d}.jsonl")

    existing_idx_global, existing_q_global, existing_map_by_idx, existing_map_by_q, total_lines = \
        scan_existing_outputs(output_dir, base_name)

    model, tok = load_model_tokenizer(args.model_name_or_path, args.trust_remote_code, device)

    sampler = DistributedSampler(
        list(range(N)), num_replicas=world_size, rank=rank,
        shuffle=False, drop_last=False
    )
    raw_idx = list(iter(sampler))
    seen = set()
    my_indices = []
    for i in raw_idx:
        if i < N and i not in seen:
            seen.add(i)
            my_indices.append(i)

    if rank == 0:
        print(f"[rank {rank}] world_size = {world_size}")
        print(f"[rank {rank}] output_dir = {output_dir}")
        print(f"[rank {rank}] combined existing lines = {total_lines}")

    print(f"[rank {rank}] shard_file = {shard_path}")
    print(f"[rank {rank}] will process {len(my_indices)} items")

    pbar = tqdm(total=len(my_indices), desc=f"rank {rank}", position=rank, leave=True)
    sys_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

    for i in my_indices:
        q = data[i]
        q_text = q.get("problem", "")
        hidden_path = os.path.join(output_dir, f"hidden_{i}.pt")

        if (i in existing_idx_global) or (q_text in existing_q_global):
            if args.save_step_hs and not os.path.exists(hidden_path):
                try:
                    entry = existing_map_by_idx.get(i) or existing_map_by_q.get(q_text)
                    if entry and entry.get("generated_responses"):
                        response_text = entry["generated_responses"][0] if entry["generated_responses"] else ""
                        prompt = tok.apply_chat_template(
                            [
                                {"role": "system", "content": sys_prompt},
                                {"role": "user", "content": q_text},
                            ],
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        prompt_ids = tok(prompt, return_tensors="pt")["input_ids"]
                        prompt_len = int(prompt_ids.shape[-1])
                        full_text = prompt + response_text
                        full_ids_cpu_1xT = tok(full_text, return_tensors="pt")["input_ids"].cpu()
                        save_double_newline_token_hs_before_think_all(
                            model=model,
                            tokenizer=tok,
                            full_ids_cpu_1xT=full_ids_cpu_1xT,
                            prompt_len=prompt_len,
                            save_path=hidden_path,
                            hs_device=args.hs_device,
                            block_size=args.hs_block_size,
                        )
                        del prompt_ids, full_ids_cpu_1xT
                        _clear_device()
                        print(f"[rank {rank}] rescued hidden for idx={i}")
                except Exception as e_rescue:
                    print(f"[WARN][rank {rank}] rescue hidden failed at idx={i}: {e_rescue}")
                    _clear_device()
            pbar.update(1)
            continue

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q_text},
        ]

        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(device)

        gen_kwargs = dict(
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            return_dict_in_generate=True,
            pad_token_id=tok.eos_token_id,
        )

        try:
            with torch.inference_mode():
                gen = model.generate(**inputs, **gen_kwargs)
        except _oom_exception_types():
            _clear_device()
            pbar.update(1)
            continue

        full_ids_1xT = gen.sequences  # [1, T] on device
        prompt_len = int(inputs["input_ids"].shape[-1])

        # decode response
        gen_ids = full_ids_1xT[0, prompt_len:].detach().cpu()
        response_text = tok.decode(gen_ids, skip_special_tokens=True)
        generate_response_length = int(gen_ids.numel())

        if args.save_step_hs:
            try:
                full_ids_cpu_1xT = full_ids_1xT.detach().cpu()
                save_double_newline_token_hs_before_think_all(
                    model=model,
                    tokenizer=tok,
                    full_ids_cpu_1xT=full_ids_cpu_1xT,
                    prompt_len=prompt_len,
                    save_path=hidden_path,
                    hs_device=args.hs_device,
                    block_size=args.hs_block_size,
                )
                del full_ids_cpu_1xT
            except _oom_exception_types():
                _clear_device()

        out_row = {
            "idx": i,
            "question": q_text,
            "generated_responses": [response_text],
            "gold_answer": q.get("answer", ""),
            "generate_response_length": generate_response_length,
        }
        append_jsonl(shard_path, out_row)

        del gen, inputs, full_ids_1xT, gen_ids
        _clear_device()
        pbar.update(1)

    pbar.close()


# ------------------------
# Main
# ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--dataset_dir", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--output_path", type=str, required=True)

    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--run_id", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)

    # ✅ defaults aligned to author settings
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_new_tokens", type=int, default=32000)

    ap.add_argument("--save_step_hs", action="store_true")
    ap.add_argument("--hs_device", type=str, default="auto", choices=["auto", "npu", "cuda", "cpu"])
    ap.add_argument("--hs_block_size", type=int, default=256)
    ap.add_argument("--keep_shards", action="store_true")

    args = ap.parse_args()

    is_dist, rank, world_size, local_rank, device = init_distributed_if_needed()

    # 打印接收到的参数
    if rank == 0:
        print("=" * 80)
        print("Received Arguments:")
        print("=" * 80)
        for arg, value in vars(args).items():
            print(f"  {arg} = {value}")
        print("=" * 80)

    worker(args, rank, world_size, device)

    if is_dist:
        dist.barrier()

    if rank == 0:
        output_dir, base_name = build_output_paths(args)
        combined_file, count = merge_all_shards(
            output_dir, base_name, remove_shards=not args.keep_shards
        )
        print(f"[rank 0] ✅ Merged {count} entries to: {combined_file}")

    if is_dist and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
