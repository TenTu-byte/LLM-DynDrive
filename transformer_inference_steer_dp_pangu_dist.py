import os
import json
import argparse
import gc
import glob
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from re import split as rsplit
import random

# NOTE: keep your custom Qwen/Pangu import path
from modeling_utils.modeling_openpangu_dynamic_3D import PanguEmbeddedForCausalLM

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
    
    # Optional run_id prefix
    if args.run_id:
        components.append(args.run_id.strip())
    
    components.append("steer")
    
    # Always include max_generated_tokens
    components.append(f"maxlen{args.max_generated_tokens}")
    
    # Always include seed
    components.append(f"seed{args.seed}")
    
    # Optional low_val_2
    if args.low_val_2 is not None:
        components.append(f"low2_{args.low_val_2}")
    
    # Optional high_val_2
    if args.high_val_2 is not None:
        components.append(f"high2_{args.high_val_2}")
    
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

    model = PanguEmbeddedForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=True
    ).to(device)

    model.eval()
    return model, tokenizer, dtype

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
    device: torch.device
):
    """
    返回：
      generated_ids: 仅新增生成部分的 token ids (Tensor[T], device)
      token_logprobs: list[float]（若不需要可忽略）
    """
    token_logprobs = []
    generated = []
    past_key_values = None

    # 先基于prompt建立KV
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, past_key_values=None)
    past_key_values = outputs.past_key_values

    last_logits = outputs.logits[:, -1, :]

    for _ in range(max_new_tokens):
        # 采样一步
        next_token, next_logprob = top_p_sampling_step(last_logits, temperature, top_p, output_logprobs=False)
        # （如需logprob，取消下一行注释）
        # token_logprobs.append(next_logprob)

        generated.append(next_token.item())
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break

        # 基于上一步 token 继续前推（复用KV）
        out = model(
            input_ids=next_token,  # [1,1]
            attention_mask=None,
            use_cache=True,
            past_key_values=past_key_values,
        )
        past_key_values = out.past_key_values
        last_logits = out.logits[:, -1, :]

    if len(generated) == 0:
        return torch.empty(0, dtype=torch.long, device=device), []

    return torch.tensor(generated, dtype=torch.long, device=device), token_logprobs

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

    # Load steer vector & enable steering
    steer_vector = torch.load(args.steer_vector_path, map_location="cpu").to(device, dtype=dtype)
    model.set_steering_flag(
        steering_flag=True,
        steering_layer=args.steer_layer,
        steer_vec=steer_vector,
        steer_coef=args.steer_coef,
        tokenizer=tokenizer,
        low_val_2=args.low_val_2,
        high_val_2=args.high_val_2
    )

    # 自动划分样本（不再手写 i % world_size）
    sampler = DistributedSampler(
        list(range(N)),
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False
    )
    sampler.set_epoch(0)
    raw_idx = list(iter(sampler))
    # 去重以防 padding
    seen, my_indices = set(), []
    for i in raw_idx:
        if i < N and i not in seen:
            seen.add(i)
            my_indices.append(i)

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

        # 自定义会话 API
        model.start_new_round()

        messages = [
            {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
            {"role": "user", "content": qtext}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        try:
            with torch.no_grad():
                gen_ids, step_logprobs = sample_with_tracking(  # TODO: actually token_logprobs
                    model=model,
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", torch.ones_like(inputs["input_ids"])),
                    max_new_tokens=args.max_generated_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    eos_token_id=tokenizer.eos_token_id,
                    dtype=dtype,
                    device=device
                )
                full_ids = torch.cat([inputs["input_ids"][0], gen_ids], dim=0)
                response_text = tokenizer.decode(full_ids[inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        except torch.npu.OutOfMemoryError:
            print(f"[OOM][rank {rank}] Skipping idx={i} : {qtext}...")
            empty_device_cache()
            pbar.update(1)
            continue
        except Exception as e:
            print(f"[ERROR][rank {rank}] idx={i} : {e}")
            pbar.update(1)
            continue

        result = {
            "idx": i,
            "question": qtext,
            "generated_responses": [response_text],
            "gold_answer": q.get("answer", "")
        }

        with open(shard_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

        del inputs, gen_ids
        empty_device_cache()
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
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--steer_vector_path', type=str, required=True)
    parser.add_argument('--steer_layer', type=int, default=27)
    parser.add_argument('--steer_coef', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.95)
    parser.add_argument('--run_id', type=str, default="", help="Optional tag to separate different runs in filenames")
    parser.add_argument('--max_generated_tokens', type=int, default=16000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--low_val_2', type=parse_optional_float, default=None)
    parser.add_argument('--high_val_2', type=parse_optional_float, default=None)
    args = parser.parse_args()
    set_seed(args.seed)

    is_dist, rank, world_size, local_rank, device = init_distributed_if_needed()

    worker(args, rank, world_size, local_rank, device)

    # 同步 & 汇总
    if is_dist:
        dist.barrier()
    if rank == 0:
        output_dir, base_name = build_output_paths(args)
        combined_file, count = merge_all_shards(output_dir, base_name, remove_shards=True)
        print(f"[rank 0] ✅ Merged {count} entries to: {combined_file}")
    if is_dist and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
