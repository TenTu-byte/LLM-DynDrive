# -*- coding: utf-8 -*-
"""
OOM-hardened multi-node data-parallel inference with hidden-state alignment fixes:
  • 多节点/多卡分布式 (torch.distributed, HCCL on NPU)
  • DistributedSampler 自动数据分发（不再手写取模）
  • 各 rank 写 shard；rank 0 自动合并汇总（去重、排序）
  • 保留你的所有逻辑（model/tokenizer加载、采样、置信度、shard格式等）
  • 修复 hidden_state 失败后 JSON 已写导致永远缺失的问题（自动救援补写，扫描全量/分片）
  • save_pangu_think_split_tokens_only CPU 兜底+设备还原+原子保存
  • 先 hidden 后 JSON，hidden 失败下次自动补写；step_hidden_num 始终有定义
"""

import warnings
warnings.filterwarnings("ignore")

from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

import os
import json
import argparse
import sys
import gc
import glob
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
from typing import List
from re import split as rsplit
from contextlib import nullcontext


# ------------------------
# Utils
# ------------------------

def write_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    tmp = file_path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    os.replace(tmp, file_path)


def read_jsonl(file_path):
    if not os.path.exists(file_path):
        print(f"Warning: Dataset file not found at {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(seed)
        # 以下两行对 NPU/Ascend 不是必须，但沿用你的设置
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass


def _clear_npu():
    if hasattr(torch, "npu") and torch.npu.is_available():
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
    gc.collect()


def _clear_cuda():
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


def _clear_device():
    _clear_npu()
    _clear_cuda()

# ------------------------
# Distributed helpers
# ------------------------

def _auto_backend():
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "hccl"   # Ascend NPU
    if torch.cuda.is_available():
        return "nccl"   # NVIDIA GPU
    return "gloo"       # CPU fallback


def init_distributed_if_needed():
    """使用 torchrun 跨节点启动时通过 env:// 初始化；否则单进程回退。"""
    # Automatically set by torchrun
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_dist = world_size > 1
    rank = 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if is_dist and not dist.is_initialized():
        backend = _auto_backend()
        dist.init_process_group(backend=backend, init_method="env://")  # use environment variables
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        world_size = 1
        rank = 0

    # 设备
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(local_rank)  # default device
        device = torch.device(f"npu:{local_rank}")
    elif torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    return is_dist, rank, world_size, local_rank, device


# ------------------------
# Confidence helpers
# ------------------------

def _summarize_selected_logprobs(selected_logprobs: torch.Tensor, policy: str = 'avg2') -> float:
    if selected_logprobs is None or selected_logprobs.numel() == 0:
        return 0.0
    probs = torch.exp(selected_logprobs)
    if policy == 'min':
        return probs.min().item()
    elif policy == 'avg1':
        return probs.mean().item()
    return torch.exp(selected_logprobs.mean()).item()


def compute_token_logprobs_streaming(model, prompt_ids: torch.Tensor, gen_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    assert prompt_ids.ndim == 2 and prompt_ids.size(0) == 1
    assert gen_ids.ndim == 1

    logps = []

    with torch.inference_mode():#, autocast_ctx:
        # 首 token
        out = model(prompt_ids.to(device), use_cache=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        logprob0 = F.log_softmax(logits, dim=-1)[0, gen_ids[0].to(device)].item()
        logps.append(logprob0)

        if gen_ids.numel() > 1:
            prev = gen_ids[0].view(1, 1).to(device)
            # TODO: discard the non-thinking sequence
            for t in range(1, gen_ids.numel()):
                out = model(prev, use_cache=True, past_key_values=past, return_dict=True)
                past = out.past_key_values
                logits = out.logits[:, -1, :]
                lp = F.log_softmax(logits, dim=-1)[0, gen_ids[t].to(device)].item()
                logps.append(lp)
                prev = gen_ids[t].view(1, 1).to(device)

    return torch.tensor(logps, dtype=torch.float32)


# ------------------------
# Model / tokenizer
# ------------------------

def load_model_and_tokenizer_single_npu(args, device):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
        local_files_only=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype="auto",
        local_files_only=True
    )

    model.to(device)
    model.eval()
    return model, tokenizer


# ------------------------
# Hidden-state saving (修复版)
# ------------------------

def save_pangu_think_split_tokens_only(model, tokenizer, input_ids, full_text, save_path, think_end_id, split_ids, hs_device: str = 'auto'):
    import os as _os

    ids_flat = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    def _find_subseq(seq, subseq):
        if not subseq:
            return None
        L, M = len(seq), len(subseq)
        for s in range(0, L - M + 1):
            if seq[s:s + M] == subseq:
                return s
        return None

    pos = _find_subseq(ids_flat, [think_end_id])
    think_end_idx = pos if pos is not None else len(ids_flat)

    step_positions = []
    split_ids_set = set(split_ids.tolist())
    for i in range(think_end_idx - 1):
        if ids_flat[i] in split_ids_set and ids_flat[i + 1] not in split_ids_set:
            step_positions.append(i + 1)

    hidden_dict = {}

    def _forward_on(device):
        with torch.inference_mode():
            outputs = model(input_ids.to(device), output_hidden_states=True, use_cache=False, return_dict=True)
        hidden_states = outputs.hidden_states
        for layer_id, layer_h in enumerate(hidden_states):
            h = layer_h.squeeze(0)
            if len(step_positions) > 0:
                idx = torch.tensor(step_positions, dtype=torch.long, device=h.device)
                step_h = h.index_select(dim=0, index=idx).to('cpu', non_blocking=True)
            else:
                step_h = torch.empty((0, h.shape[1]), dtype=h.dtype)
            hidden_dict[layer_id] = step_h

    orig_device = next(model.parameters()).device
    try:
        if hs_device in ('auto', 'npu') and hasattr(torch, "npu") and torch.npu.is_available():
            _forward_on(orig_device)
        elif hs_device in ('auto', 'cuda') and torch.cuda.is_available():
            _forward_on(orig_device)
        else:
            if orig_device.type != 'cpu':
                model.to('cpu')
            _forward_on(torch.device('cpu'))
    except torch.npu.OutOfMemoryError:
        print("[hs][npu OOM] falling back to CPU for hidden-state dump...")
        if hasattr(torch, "npu") and torch.npu.is_available():
            _clear_npu()
        if hs_device == 'npu' or 'cuda':
            raise
        model.to('cpu')
        _forward_on(torch.device('cpu'))
    # except torch.cuda.OutOfMemoryError:
    #     print("[hs][cuda OOM] falling back to CPU for hidden-state dump...")
    #     if torch.cuda.is_available():
    #         _clear_cuda()
    #     if hs_device == 'npu' or 'cuda':
    #         raise
    #     model.to('cpu')
    #     _forward_on(torch.device('cpu'))
    finally:
        if next(model.parameters()).device != orig_device:
            model.to(orig_device)

    _os.makedirs(_os.path.dirname(save_path), exist_ok=True)
    tmp_path = save_path + ".tmp"
    torch.save(hidden_dict, tmp_path)
    _os.replace(tmp_path, save_path)
    return len(step_positions)


# ------------------------
# Shard helpers
# ------------------------

def load_existing_indices(file_path):
    existing_idx = set()
    existing_q = set()
    num_lines = 0
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                num_lines += 1
                try:
                    obj = json.loads(s)
                    if "idx" in obj:
                        existing_idx.add(int(obj["idx"]))
                    if "question" in obj:
                        existing_q.add(obj["question"])
                except Exception:
                    continue
    return existing_idx, existing_q, num_lines


def _read_jsonl_map_by_idx(file_path):
    m = {}
    if not os.path.exists(file_path):
        return m
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if "idx" in obj:
                    m[int(obj["idx"])] = obj
            except Exception:
                continue
    return m


def scan_existing_outputs(output_dir, base_name):
    """
    扫描总表 + 各分片, 汇总已有的 idx 与 question, 构建 idx->obj / q->obj。
    便于：重复运行时的 hidden 救援、重复样本跳过、最终合并去重。
    """
    combined = os.path.join(output_dir, f'{base_name}.jsonl')
    shard_glob = os.path.join(output_dir, f'{base_name}.shard*.jsonl')

    files = []
    if os.path.exists(combined):
        files.append(combined)
    files += sorted(glob.glob(shard_glob))

    idx_set, q_set = set(), set()
    idx_map, q_map = {}, {}
    total = 0

    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
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
        {"role": "user", "content": question_text}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + (response_text or "")


def merge_all_shards(output_dir, base_name, remove_shards=True):
    """rank 0 在所有进程结束后调用：合并 *.shard*.jsonl (+ 旧总表若存在)，按 idx 去重并排序，写为 {base}.jsonl"""
    shard_files = sorted(glob.glob(os.path.join(output_dir, f'{base_name}.shard*.jsonl')))
    combined_file = os.path.join(output_dir, f'{base_name}.jsonl')

    # 也将旧 combined 合并，保证断点恢复
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

    # 按 idx 排序输出
    final = [merged[k] for k in sorted(merged.keys())]
    write_jsonl(final, combined_file)

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
# Worker
# ------------------------

def worker(args, rank, world_size, device):
    # 读取数据
    dataset_file = 'train.jsonl' if args.dataset == 'Math_Math' else 'test.jsonl'
    dataset_path = os.path.join(args.dataset_dir, args.dataset, dataset_file)
    questions = read_jsonl(dataset_path)
    if args.dataset == 'Math_Math':
        questions = questions[:500]
    N = len(questions)  # TODO

    model_basename = os.path.basename(os.path.normpath(args.model_name_or_path))
    output_dir = os.path.join(args.output_path, model_basename, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    base_name = f'origin_temp{args.temperature}_maxlen{args.max_generated_tokens}'
    shard_file = os.path.join(output_dir, f'{base_name}.shard{rank:03d}.jsonl')

    # 扫描现有结果（总表 + 全部 shard），用于跳过/救援
    existing_idx_global, existing_q_global, existing_map_by_idx, existing_map_by_q, total_lines = \
        scan_existing_outputs(output_dir, base_name)

    # 模型
    model, tokenizer = load_model_and_tokenizer_single_npu(args, device)

    vocab = tokenizer.get_vocab()
    split_ids = torch.LongTensor([vocab[token] for token in vocab.keys() if "\n\n" in token]).to(device)
    think_start_id = tokenizer.encode("[unused16]", add_special_tokens=False)[0]  # int
    think_end_id =  tokenizer.encode("[unused17]", add_special_tokens=False)[0]

    # 使用 DistributedSampler 自动划分索引（不再手写取模）
    sampler = DistributedSampler(  # rank=i => indices[i:total_size:world_size]
        list(range(N)), num_replicas=world_size, rank=rank,
        shuffle=False, drop_last=False
    )

    # 取 sampler 的分配结果并去除 padding/重复
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

    print(f"[rank {rank}] shard_file = {shard_file}")
    print(f"[rank {rank}] will process {len(my_indices)} items")

    pbar = tqdm(total=len(my_indices), desc=f"Rank {rank} DP Inference", position=rank, leave=True)
    sys_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

    for i in my_indices:
        q = questions[i]
        q_text = q.get("problem", "")
        hidden_save_path = os.path.join(output_dir, f"hidden_{i:03}.pt")

        # 若该样本已在历史结果里出现，且缺 hidden，则尝试救援（从历史响应复原 full_text 再导出 hidden）
        if (i in existing_idx_global) or (q_text in existing_q_global):
            if not os.path.exists(hidden_save_path):
                try:
                    entry = existing_map_by_idx.get(i) or existing_map_by_q.get(q_text)
                    if entry and entry.get("generated_responses"):
                        response_text = entry["generated_responses"][0] if entry["generated_responses"] else ""
                        full_text = _reconstruct_full_text(tokenizer, sys_prompt, q_text, response_text)
                        full_inputs = tokenizer(full_text, return_tensors="pt")
                        _ = save_pangu_think_split_tokens_only(
                            model, tokenizer, full_inputs.input_ids.to(device), full_text, hidden_save_path,
                            think_end_id,
                            split_ids,
                            hs_device=args.hs_device,
                        )
                        del full_inputs
                        print(f"[rank {rank}] rescued hidden for idx={i}")
                        _clear_device
                except Exception as e_rescue:
                    print(f"[WARN][rank {rank}] rescue hidden failed at idx={i}: {e_rescue}")
                    _clear_device
            pbar.update(1)
            continue

        # prompt 构造与推理
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q_text}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # ! With `return_tensors="pt"`, `input_ids` are always 2D, brackets or not;
        # ! without it, they're 1D without brackets, 2D with brackets.
        inputs = tokenizer([prompt], return_tensors="pt").to(device)

        try:
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_generated_tokens,
                    return_dict_in_generate=True,
                    eos_token_id=tokenizer.eos_token_id
                )
        except torch.npu.OutOfMemoryError:
            print(f"[OOM][rank {rank}] idx={i} : {q_text[:80]}... skipping.")
            _clear_device()
            pbar.update(1)
            continue
        except Exception as e:
            print(f"[ERROR][rank {rank}] idx={i} : {e}")
            pbar.update(1)
            continue

        response_text = tokenizer.decode(
            output.sequences[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        full_text = prompt + response_text

        # 逐 token logprob
        gen_token_ids = output.sequences[0][inputs.input_ids.shape[1]:].detach().cpu()
        try:
            gen_logps = compute_token_logprobs_streaming(
                model,
                prompt_ids=inputs.input_ids,
                gen_ids=gen_token_ids,
                device=device,
            )
        except torch.npu.OutOfMemoryError:
            print(f"[OOM][rank {rank}] logprob pass at idx={i}; skipping confidences.")
            gen_logps = torch.empty(0)
        except Exception as e:
            print(f"[WARN][rank {rank}] logprob pass failed at idx={i}: {e}")
            gen_logps = torch.empty(0)

        # 句级置信度（使用 split_ids 和 think_end_id 切段）
        gen_token_ids_list = gen_token_ids.tolist()
        split_ids_set = set(split_ids.tolist())
        
        # 找到 think_end 位置
        try:
            think_end_pos = gen_token_ids_list.index(think_end_id)
        except ValueError:
            think_end_pos = len(gen_token_ids_list)
        
        # 在 think_end 之前，根据 split_ids 划分段落
        step_positions = []
        for i in range(think_end_pos - 1):
            # 当前 token 是分隔符，且下一个 token 不是分隔符时，标记为段落起始
            if gen_token_ids_list[i] in split_ids_set and gen_token_ids_list[i + 1] not in split_ids_set:
                step_positions.append(i + 1)

        # 计算每个段落的置信度
        confidences = []
        for idx, start_pos in enumerate(step_positions):
            # 段落结束位置：下一个段落的起始位置，或 think_end
            end_pos = step_positions[idx + 1] if idx + 1 < len(step_positions) else think_end_pos
            
            if gen_logps.numel() > 0 and start_pos < end_pos and end_pos <= gen_logps.numel():
                seg_logps = gen_logps[start_pos:end_pos]
                conf = _summarize_selected_logprobs(seg_logps, policy='avg2')
                confidences.append(conf)
            elif start_pos < end_pos:
                confidences.append(0.0)

        # 先保存 hidden，再写 JSON 行（并保证 step_hidden_num 一定有值）
        step_hidden_num = -1
        try:
            if not os.path.exists(hidden_save_path):
                full_inputs = tokenizer(full_text, return_tensors="pt")
                step_hidden_num = save_pangu_think_split_tokens_only(
                    model, tokenizer, full_inputs.input_ids.to(device), full_text, hidden_save_path,
                    think_end_id,
                    split_ids,
                    hs_device=args.hs_device
                )
                del full_inputs
            else:
                print(f"[rank {rank}] hidden exists, skip: {hidden_save_path}")
        except Exception as he:
            print(f"[WARN][rank {rank}] hidden save failed at idx={i}: {he} (rescueable next run)")
            _clear_device()

        # 写本 rank 的 shard 行
        result_entry = {
            "idx": i,
            "question": q_text,
            "generated_responses": [response_text],
            "gold_answer": q.get("answer", ""),
            "sentence_confidences": confidences,
            "confidence_num": len(confidences),
            "step_hidden_num": step_hidden_num
        }
        with open(shard_file, 'a', encoding='utf-8') as fout:
            fout.write(json.dumps(result_entry, ensure_ascii=False) + '\n')

        del output, inputs, gen_token_ids, gen_logps
        _clear_device()
        pbar.update(1)

    pbar.close()
    print(f"[rank {rank}] ✅ Done. Shard saved to {shard_file}")


# ------------------------
# Main
# ------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--max_generated_tokens', type=int, default=512)
    parser.add_argument('--trust_remote_code', action='store_true')
    parser.add_argument('--hs_device', type=str, default='auto', choices=['auto', 'npu', 'cpu'])
    parser.add_argument('--score_dtype', type=str, default='bf16', choices=['bf16', 'fp16', 'fp32'])
    args = parser.parse_args()
    set_seeds(42)

    is_dist, rank, world_size, local_rank, device = init_distributed_if_needed()

    # 每个进程各自跑 worker
    worker(args, rank, world_size, device)

    # 所有进程同步；rank 0 合并汇总
    if is_dist:
        dist.barrier()

    if rank == 0:
        model_basename = os.path.basename(os.path.normpath(args.model_name_or_path))
        output_dir = os.path.join(args.output_path, model_basename, args.dataset)
        base_name = f'origin_temp{args.temperature}_maxlen{args.max_generated_tokens}'
        combined_file, count = merge_all_shards(output_dir, base_name, remove_shards=True)
        print(f"[rank 0] ✅ Merged {count} entries to: {combined_file}")

    if is_dist and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
