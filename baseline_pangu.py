# -*- coding: utf-8 -*-
"""
openPangu baseline evaluation
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
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
from typing import List
from re import split as rsplit


# ------------------------
# Utils
# ------------------------

def write_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


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
    if torch.npu.is_available():
        torch.npu.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("high")


def _clear_npu():
    if torch.npu.is_available():
        try:
            torch.npu.synchronize()
        except Exception:
            pass
        torch.npu.empty_cache()
        try:
            torch.npu.ipc_collect()  # Inter-Process Communication
        except Exception:
            pass
    gc.collect()  # Garbage Collection


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


def compute_token_logprobs_streaming(model, tokenizer, prompt_ids: torch.Tensor, gen_ids: torch.Tensor,
                                     device: torch.device, score_dtype: str = 'bf16') -> torch.Tensor:
    assert prompt_ids.ndim == 2 and prompt_ids.size(0) == 1
    assert gen_ids.ndim == 1

    dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
    amp_dtype = dtype_map.get(score_dtype, torch.bfloat16)

    logps = []
    with torch.inference_mode(), torch.npu.amp.autocast(enabled=(device.type == 'npu'), dtype=amp_dtype):
        out = model(prompt_ids.to(device), use_cache=True, return_dict=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        # TODO: what is calculated here is the probability of the sampled token, rather than the maximum probability
        logprob0 = F.log_softmax(logits, dim=-1)[0, gen_ids[0].to(device)].item()
        logps.append(logprob0)

        if gen_ids.numel() > 1:
            prev = gen_ids[0].view(1, 1).to(device)
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

    dtype = torch.float16 if torch.npu.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True
    )

    model.to(device)
    model.eval()

    return model, tokenizer, dtype


# ------------------------
# Hidden-state saving (修复版)
# ------------------------

def save_pangu_think_split_tokens_only(model, tokenizer, input_ids, full_text, save_path,
                                       hs_device: str = 'auto'):
    import os as _os

    ids_flat = input_ids[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])  # TODO: change to tokenizer.decode()?
    think_ids = tokenizer.encode("[unused17]", add_special_tokens=False)  # ! adapt to pangu

    def _find_subseq(seq, subseq):
        if not subseq:
            return None
        L, M = len(seq), len(subseq)
        for s in range(0, L - M + 1):
            if seq[s:s + M] == subseq:
                return s
        return None

    pos = _find_subseq(ids_flat, think_ids)
    think_end_idx = pos if pos is not None else len(ids_flat)

    step_positions = []
    for i in range(think_end_idx - 1):
        tok = tokens[i]
        next_tok = tokens[i + 1] if i + 1 < think_end_idx - 1 else ""  # using think_end_idx - 1 here allows next_tok to be empty
        if "\n\n" in tok and "\n\n" not in next_tok:  # ! adpat to pangu
            step_positions.append(i + 1)

    hidden_dict = {}

    def _forward_on(device):
        with torch.inference_mode():
            outputs = model(input_ids.to(device), output_hidden_states=True, use_cache=False, return_dict=True)
        hidden_states = outputs.hidden_states
        for layer_id, layer_h in enumerate(hidden_states):
            h = layer_h.squeeze(0)
            if len(step_positions) > 0:
                idx = torch.tensor(step_positions, dtype=torch.long, device=h.device)  # TODO: extract the next token of /n/n?
                step_h = h.index_select(dim=0, index=idx).to('cpu', non_blocking=True)
            else:
                step_h = torch.empty((0, h.shape[1]), dtype=h.dtype)
            hidden_dict[layer_id] = step_h  # simplify the dict structure

    orig_device = next(model.parameters()).device
    try:
        if hs_device in ('auto', 'npu') and torch.npu.is_available():
            _forward_on(orig_device)
        else:
            if orig_device.type != 'cpu':
                model.to('cpu')
            _forward_on(torch.device('cpu'))
    except torch.npu.OutOfMemoryError:
        print("[hs][npu OOM] falling back to CPU for hidden-state dump...")
        _clear_npu()  # TODO: clean up NPU cache here actually doesn't have much effect?
        if hs_device == 'npu':
            raise
        model.to('cpu')
        _forward_on(torch.device('cpu'))
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

def load_existing_indices(shard_file):
    existing_idx = set()
    existing_q = set()
    num_lines = 0
    if os.path.exists(shard_file):
        with open(shard_file, 'r', encoding='utf-8') as f:
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


def _reconstruct_full_text(tokenizer, sys_prompt, question_text, response_text):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question_text}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt + (response_text or "")


# ------------------------
# Worker
# ------------------------

def worker(rank, world_size, args):
    # set_seeds(42 + rank)
    if torch.npu.is_available():
        torch.npu.set_device(rank)
    device = torch.device(f"npu:{rank}" if torch.npu.is_available() else "cpu")

    # Dataset
    dataset_path = os.path.join(args.dataset_dir, args.dataset, 'test.jsonl')
    questions = read_jsonl(dataset_path)

    # Output paths
    model_basename = os.path.basename(os.path.normpath(args.model_name_or_path))
    output_dir = os.path.join(args.output_path, model_basename, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    base_name = f'origin_temp{args.temperature}_maxlen{args.max_generated_tokens}'
    shard_file = os.path.join(output_dir, f'{base_name}.shard{rank:03d}.jsonl')

    # Resume from checkpoint
    existing_idx, existing_q, num_lines = load_existing_indices(shard_file)

    # Load model/tokenizer per rank
    model, tokenizer, dtype = load_model_and_tokenizer_single_npu(args, device)  # unify the dtype

    # Partition: each rank handles indices congruent to its rank
    my_indices = [i for i in range(len(questions)) if (i % world_size) == rank]

    # Logging
    print(f"[rank {rank}] world_size={world_size}")
    print(f"[rank {rank}] shard_file = {shard_file}")
    print(f"[rank {rank}] loaded existing: idx={len(existing_idx)}, question={len(existing_q)}, lines={num_lines}")
    print(f"[rank {rank}] will process {len(my_indices)} items")

    pbar = tqdm(total=len(my_indices), desc=f"Rank {rank} DP Inference", position=rank, leave=True)

    sys_prompt = "Please reason step by step, and put your final answer within \\boxed{}."  # unify prompt here

    for i in my_indices:
        q = questions[i]

        if (i in existing_idx) or (q.get("problem") in existing_q):
            pbar.update(1)
            continue

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q['problem']}
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
            _clear_npu()
            print(f"[OOM][rank {rank}] idx={i} : {q['problem'][:80]}... skipping.")
            pbar.update(1)
            continue
        except Exception as e:
            print(f"[ERROR][rank {rank}] idx={i} : {e}")
            pbar.update(1)
            continue

        response_text = tokenizer.decode(  # str
            output.sequences[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        result = {
            "idx": i,
            "question": q.get("problem", ""),
            "generated_responses": [response_text],
            "gold_answer": q.get("answer", ""),
        }

        # Append to shard files
        with open(shard_file, 'a', encoding='utf-8') as fout:
            fout.write(json.dumps(result, ensure_ascii=False) + '\n')

        del output, inputs
        _clear_npu()
        pbar.update(1)

    pbar.close()
    print(f"[rank {rank}] ✅ Done. Results saved to {shard_file}")


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
    parser.add_argument('--num_npus', type=int, default=1)
    parser.add_argument('--hs_device', type=str, default='auto', choices=['auto', 'npu', 'cpu'])
    parser.add_argument('--score_dtype', type=str, default='bf16', choices=['bf16', 'fp16', 'fp32'])
    args = parser.parse_args()

    world_size = max(1, int(args.num_npus))
    if world_size == 1:
        worker(0, 1, args)
    else:
        mp.spawn(worker, nprocs=world_size, args=(world_size, args))


if __name__ == "__main__":
    main()
