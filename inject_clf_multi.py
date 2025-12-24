# -*- coding: utf-8 -*-
import os
import json
import argparse
import gc
import random
import numpy as np
import torch
import torch.multiprocessing as mp

import joblib

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList,
)
from tqdm import tqdm


# ------------------------
# Small utils
# ------------------------

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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


def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_done_indices(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if "idx" in obj:
                    done.add(int(obj["idx"]))
            except Exception:
                continue
    return done


# ------------------------
# Stop-id builder (tokens containing "\n\n")
# ------------------------

def build_stop_ids_contains(tok, needle: str = "\n\n"):
    vocab = tok.get_vocab()  # token_str -> id
    stop_ids = {int(i) for s, i in vocab.items() if needle in s}
    if not stop_ids:
        raise RuntimeError(f'No tokens containing "{needle}" found in tokenizer vocab.')
    return stop_ids


# ------------------------
# Stop criteria: stop when last generated token is in stop_ids
# ------------------------

class StopOnTokenIdSet(StoppingCriteria):
    def __init__(self, stop_ids, prompt_len: int):
        super().__init__()
        self.stop_ids = set(int(x) for x in stop_ids)
        self.prompt_len = int(prompt_len)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        T = int(input_ids.shape[-1])
        if T <= self.prompt_len:
            return False
        return int(input_ids[0, -1]) in self.stop_ids


# ------------------------
# Model load
# ------------------------

def load_model_tokenizer(model_name_or_path, trust_remote_code, device):
    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch.bfloat16 if (torch.cuda.is_available() and device.type == "cuda") else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model, tok


# ------------------------
# Save hidden states at FIRST stop token only (compat with your previous pipeline)
# ------------------------

def first_index_in_set(seq, id_set):
    s = set(int(x) for x in id_set)
    for t, tid in enumerate(seq):
        if int(tid) in s:
            return t
    return None


def save_double_newline_token_hs_first_only(
    model,
    tokenizer,
    full_ids_1xT: torch.Tensor,   # [1, T] on device
    prompt_len: int,
    save_path: str,
    hs_device: str = "auto",
):
    import os as _os
    stop_ids = build_stop_ids_contains(tokenizer, needle="\n\n")

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
        if hs_device in ("auto", "cuda") and torch.cuda.is_available():
            _forward_on(orig_device)
        else:
            if orig_device.type != "cpu":
                model.to("cpu")
            _forward_on(torch.device("cpu"))
    except torch.cuda.OutOfMemoryError:
        if hs_device == "cuda":
            raise
        _clear_cuda()
        model.to("cpu")
        _forward_on(torch.device("cpu"))
    finally:
        if next(model.parameters()).device != orig_device:
            model.to(orig_device)

    _os.makedirs(_os.path.dirname(save_path), exist_ok=True)
    torch.save({"pos_full": pos_full, "pattern_ids": ([] if hit_id is None else [hit_id]), "hs": hs_out}, save_path)


# ------------------------
# Classifier feature extraction at a given position pos_full
# ------------------------

def extract_feature_at_pos(
    model,
    full_ids_1xT: torch.Tensor,   # [1, T] on device
    hs_device: str,
    layer: int,
    pos_full: int,
):
    """
    Return: rep np.float32 [H] on CPU, or None if fail
    """
    rep = None

    def _forward_on(device):
        nonlocal rep
        with torch.inference_mode():
            out = model(full_ids_1xT.to(device), output_hidden_states=True, use_cache=False, return_dict=True)
        hs = out.hidden_states
        if hs is None or len(hs) == 0:
            rep = None
            return
        L = int(layer)
        if L < 0:
            L = len(hs) - 1
        if L >= len(hs):
            L = len(hs) - 1
        h = hs[L]  # [1, T, H]
        v = h[0, int(pos_full), :].detach().to("cpu", dtype=torch.float32)
        rep = v.numpy()

    orig_device = next(model.parameters()).device
    try:
        if hs_device in ("auto", "cuda") and torch.cuda.is_available():
            _forward_on(orig_device)
        else:
            if orig_device.type != "cpu":
                model.to("cpu")
            _forward_on(torch.device("cpu"))
    except torch.cuda.OutOfMemoryError:
        if hs_device == "cuda":
            return None
        _clear_cuda()
        try:
            model.to("cpu")
            _forward_on(torch.device("cpu"))
        except Exception:
            return None
    finally:
        if next(model.parameters()).device != orig_device:
            model.to(orig_device)

    if rep is None:
        return None
    return rep.astype(np.float32)


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


# ------------------------
# Worker
# ------------------------

def worker(rank, world_size, args):
    set_seeds(int(args.seed) + rank)

    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    dataset_path = os.path.join(args.dataset_dir, args.dataset, "test.jsonl")
    data = read_jsonl(dataset_path)

    model_name = os.path.basename(os.path.normpath(args.model_name_or_path))
    out_dir = os.path.join(args.output_path, model_name, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    shard_path = os.path.join(out_dir, f"results.shard{rank}.jsonl")
    done = load_done_indices(shard_path)

    model, tok = load_model_tokenizer(args.model_name_or_path, args.trust_remote_code, device)
    stop_ids = build_stop_ids_contains(tok, needle="\n\n")

    # -------- load sklearn classifier (CPU) --------
    insert_clf = None
    meta = None
    if args.clf and args.meta:
        insert_clf = joblib.load(args.clf)
        with open(args.meta, "r", encoding="utf-8") as f:
            meta = json.load(f)

    meta_best_layer = int(meta["best_layer"]) if meta else -1
    meta_thr = float(meta.get("prob_threshold", 0.5)) if meta else 0.5
    meta_pos_is_short = bool(meta.get("pos_is_short", True)) if meta else True
    meta_n = int(meta.get("n", 0)) if meta else 0

    clf_layer = int(args.clf_layer) if args.clf_layer is not None else meta_best_layer
    clf_thr = float(args.clf_prob_threshold) if args.clf_prob_threshold is not None else meta_thr
    # default: pred1 => insert (与你之前一致)
    insert_on_pred1 = True if args.insert_on_pred1 else True

    indices = [i for i in range(len(data)) if (i % world_size) == rank]
    pbar = tqdm(total=len(indices), desc=f"rank {rank}", position=rank, leave=True)

    sys_prompt = "Please reason step by step, and put your final answer within \\boxed{}."

    for i in indices:
        if i in done:
            pbar.update(1)
            continue

        q = data[i]
        q_text = q.get("problem", "")

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": q_text},
        ]

        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(device)
        prompt_len0 = int(inputs["input_ids"].shape[-1])

        def _generate(_input_ids: torch.Tensor, max_new_tokens: int, stopping):
            _kwargs = dict(
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=int(max_new_tokens),
                return_dict_in_generate=True,
                pad_token_id=tok.eos_token_id,
            )
            if stopping is not None:
                _kwargs["stopping_criteria"] = stopping

            attn = torch.ones_like(_input_ids, device=_input_ids.device)
            with torch.inference_mode():
                return model.generate(input_ids=_input_ids, attention_mask=attn, **_kwargs)

        # --------------------------
        # Multi-checkpoint loop
        # --------------------------
        remaining = int(args.max_new_tokens)
        full_ids = inputs["input_ids"]  # [1, T]
        inserted = False
        insert_step = None
        checkpoints = []
        saved_first_hs = False

        try:
            while remaining > 0:
                cur_len = int(full_ids.shape[-1])
                stopping = StoppingCriteriaList([StopOnTokenIdSet(stop_ids, prompt_len=cur_len)])
                gen = _generate(full_ids, remaining, stopping)
                seq = gen.sequences
                new_len = int(seq.shape[-1] - cur_len)
                if new_len <= 0:
                    full_ids = seq
                    break

                remaining -= new_len
                full_ids = seq

                last_id = int(full_ids[0, -1])

                # If didn't stop on "\n\n" token, we are likely done (EOS or budget)
                if last_id not in stop_ids:
                    break

                # we hit a "\n\n" checkpoint: decide whether to insert
                pos_full = int(full_ids.shape[-1] - 1)
                ck = {
                    "step": len(checkpoints),
                    "pos_full": pos_full,
                    "last_token_id": last_id,
                    "remaining_after": int(remaining),
                    "clf_used": False,
                    "clf_proba": None,
                    "clf_pred": None,
                    "do_insert": False,
                    "reason": None,
                }

                do_insert = False
                if (not inserted) and (insert_clf is not None) and (meta is not None):
                    rep = extract_feature_at_pos(
                        model=model,
                        full_ids_1xT=full_ids,
                        hs_device=args.hs_device,
                        layer=clf_layer,
                        pos_full=pos_full,
                    )
                    if rep is not None:
                        proba = safe_predict_proba_1(insert_clf, rep)
                        pred1 = (proba >= clf_thr)
                        do_insert = bool(pred1) if insert_on_pred1 else (not bool(pred1))
                        ck.update({
                            "clf_used": True,
                            "clf_proba": float(proba),
                            "clf_pred": int(pred1),
                            "do_insert": bool(do_insert),
                            "reason": "clf_ok",
                        })
                    else:
                        ck["reason"] = "clf_rep_none"
                else:
                    ck["reason"] = "no_clf_or_already_inserted"

                checkpoints.append(ck)

                # Save FIRST checkpoint hidden file (same as old pipeline)
                if args.save_step_hs and (not saved_first_hs):
                    hidden_path = os.path.join(out_dir, f"hidden_{i}.pt")
                    try:
                        save_double_newline_token_hs_first_only(
                            model=model,
                            tokenizer=tok,
                            full_ids_1xT=full_ids,
                            prompt_len=prompt_len0,
                            save_path=hidden_path,
                            hs_device=args.hs_device,
                        )
                        saved_first_hs = True
                    except torch.cuda.OutOfMemoryError:
                        _clear_cuda()

                if do_insert and (args.insert_text or ""):
                    insert_ids = tok(args.insert_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
                    full_ids = torch.cat([full_ids, insert_ids], dim=1)
                    inserted = True
                    insert_step = ck["step"]

                    # after insertion: directly generate to end (NO further checkpoint checks)
                    if remaining > 0:
                        gen_tail = _generate(full_ids, remaining, stopping=None)
                        full_ids = gen_tail.sequences
                    break

            # end while
        except torch.cuda.OutOfMemoryError:
            _clear_cuda()
            pbar.update(1)
            continue

        # decode final
        gen_all_ids = full_ids[0, prompt_len0:].detach().cpu()
        response_text = tok.decode(gen_all_ids, skip_special_tokens=True)
        generate_response_length = int(gen_all_ids.numel())

        out_row = {
            "idx": i,
            "question": q_text,
            "generated_responses": [response_text],
            "gold_answer": q.get("answer", ""),
            "generate_response_length": generate_response_length,

            "inserted": bool(inserted),
            "insert_step": insert_step,
            "insert_policy": "clf_multicheck" if (insert_clf is not None and meta is not None) else f"fallback_{args.fallback}",
            "clf_prob_threshold": float(clf_thr),
            "clf_layer": int(clf_layer),
            "clf_pos_is_short_meta": bool(meta_pos_is_short),
            "clf_meta_n": int(meta_n),

            "checkpoints": checkpoints,
        }
        append_jsonl(shard_path, out_row)
        done.add(i)

        del full_ids, gen_all_ids, inputs
        _clear_cuda()
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

    ap.add_argument("--num_gpus", type=int, default=1)
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--min_p", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=32000)

    ap.add_argument("--insert_text", type=str, default="\n[unused17]\n\n")
    ap.add_argument("--insert_prob", type=float, default=0.5)  # kept for compatibility; not used in this multicheck path

    ap.add_argument("--save_step_hs", action="store_true")
    ap.add_argument("--hs_device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    # ---- classifier args ----
    ap.add_argument("--clf", type=str, default="", help="Path to insert_clf.joblib")
    ap.add_argument("--meta", type=str, default="", help="Path to insert_clf_meta.json")
    ap.add_argument("--clf_layer", type=int, default=None, help="Override meta.best_layer (default: use meta)")
    ap.add_argument("--clf_prob_threshold", type=float, default=None, help="Override meta.prob_threshold")
    ap.add_argument("--insert_on_pred1", action="store_true",
                    help="(kept) pred1=>insert. Default behavior is pred1=>insert even if not set.")

    # kept for compatibility with your old scripts; in this multicheck file we basically don't need fallback
    ap.add_argument("--fallback", type=str, default="never", choices=["random", "never", "always"])

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    world_size = max(1, int(args.num_gpus))
    if world_size == 1:
        worker(0, 1, args)
    else:
        mp.spawn(worker, nprocs=world_size, args=(world_size, args))


if __name__ == "__main__":
    main()
