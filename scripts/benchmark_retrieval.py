#!/usr/bin/env python3
"""Reproducible Retrieval, Embedding, and Cross-Encoder Reranking Benchmark on TPU v7x (TorchTPU).

Supports two serving paths on Google Cloud TPU v7x:
1. Native `torch.compile(backend="tpu")` in bf16 on a single TPU v7x core:
   - Bidirectional encoder models (`BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`)
   - Causal LLM embedding backbones (`Qwen/Qwen3-Embedding-8B`)
2. OpenAI-compatible `vllm serve --runner pooling` (`vllm-torchtpu v0.33.0`):
   - Live HTTP `/v1/embeddings` (`Qwen/Qwen3-Embedding-8B`)
   - Live HTTP `/v1/rerank` (`Qwen/Qwen3-Reranker-8B` with Qwen3 yes/no classifier head)

Every run includes automatic semantic quality verification (cosine similarity separation
for embeddings; top-1 relevance separation for rerankers) and finite-value checks.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time
import urllib.request


def cos_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / max(1e-12, na * nb)


def run_native_compiled(
    model_path: str,
    model_name: str,
    task: str,
    seq_lens: list[int],
    batch_sizes: list[int],
    iters: int,
    out_path: Path | None,
) -> list[dict]:
    import torch
    import torch.nn.functional as F
    import torch_tpu  # noqa: F401
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("tpu:0")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if task == "reranker":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True
        ).to(device).eval()
    else:
        model = AutoModel.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True
        ).to(device).eval()
    torch.accelerator.synchronize()
    load_s = round(time.perf_counter() - t0, 2)

    is_causal_emb = "qwen" in model_name.lower() and task == "embedding"

    def forward_fn(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if task == "reranker":
            return out.logits
        hs = out.last_hidden_state
        if is_causal_emb:
            return F.normalize(hs[:, -1, :], p=2, dim=-1)
        return F.normalize(hs[:, 0, :], p=2, dim=-1)

    compiled_fn = torch.compile(forward_fn, backend="tpu")

    # Semantic / ranking quality verification
    quality: dict = {}
    with torch.no_grad():
        if task == "embedding":
            texts = [
                "How do systolic array matrix units accelerate dense neural networks?",
                "Systolic array matrix engines execute high-throughput bf16 tensor contractions for deep learning.",
                "Classic Italian tiramisu layers espresso-soaked ladyfingers with mascarpone cream.",
            ]
            enc = tok(texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
            vecs = compiled_fn(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            torch.accelerator.synchronize()
            v = vecs.float().cpu()
            sim_pos = round(float(torch.dot(v[0], v[1])), 4)
            sim_neg = round(float(torch.dot(v[0], v[2])), 4)
            quality = {
                "dim": int(v.shape[-1]),
                "sim_pos": sim_pos,
                "sim_neg": sim_neg,
                "valid_semantic": bool(sim_pos > sim_neg + 0.15),
            }
        else:
            pairs = [
                ("How do systolic array matrix units accelerate dense neural networks?",
                 "Systolic array matrix engines execute high-throughput bf16 tensor contractions for deep learning."),
                ("How do systolic array matrix units accelerate dense neural networks?",
                 "Classic Italian tiramisu layers espresso-soaked ladyfingers with mascarpone cream."),
            ]
            enc = tok(
                [p[0] for p in pairs],
                [p[1] for p in pairs],
                padding="max_length",
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            scores = compiled_fn(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            torch.accelerator.synchronize()
            s = scores.float().cpu().reshape(-1).tolist()
            quality = {
                "score_pos": round(s[0], 4),
                "score_neg": round(s[1], 4),
                "valid_ranking": bool(s[0] > s[1] + 1.0),
            }

    records: list[dict] = []
    vocab_lim = min(10000, int(getattr(model.config, "vocab_size", 32000)) - 10)
    with torch.no_grad():
        for seq_len in seq_lens:
            for bs in batch_sizes:
                ids = torch.randint(10, vocab_lim, (bs, seq_len), dtype=torch.long, device=device)
                mask = torch.ones((bs, seq_len), dtype=torch.long, device=device)
                tc0 = time.perf_counter()
                out = compiled_fn(ids, mask)
                torch.accelerator.synchronize()
                cold_ms = (time.perf_counter() - tc0) * 1000.0

                for _ in range(3):
                    out = compiled_fn(ids, mask)
                    torch.accelerator.synchronize()

                lats = []
                for _ in range(iters):
                    ts = time.perf_counter()
                    out = compiled_fn(ids, mask)
                    torch.accelerator.synchronize()
                    lats.append((time.perf_counter() - ts) * 1000.0)
                lats.sort()
                p50 = lats[len(lats) // 2]
                p95 = lats[int(0.95 * (len(lats) - 1))]
                finite = bool(torch.isfinite(out.float().cpu()).all())
                items_s = bs / (p50 / 1000.0)
                toks_s = items_s * seq_len
                rec = {
                    "category": task,
                    "model": model_name,
                    "engine": "torchtpu-compile-bf16",
                    "chip": "v7x",
                    "cores": 1,
                    "seq_len": seq_len,
                    "batch_size": bs,
                    "load_s": load_s,
                    "cold_ms": round(cold_ms, 2),
                    "p50_ms": round(p50, 2),
                    "p95_ms": round(p95, 2),
                    "items_per_sec": round(items_s, 2),
                    "tokens_per_sec": round(toks_s, 1),
                    "finite": finite,
                    **quality,
                    "status": "ok",
                }
                print(json.dumps(rec), flush=True)
                records.append(rec)
                if out_path:
                    with out_path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
    return records


def run_vllm_rerank(base_url: str, model_name: str, tp: int, out_path: Path | None) -> list[dict]:
    query = "How do TPU systolic arrays accelerate bf16 matrix multiplication?"
    pos_doc = "TPU MXUs use systolic arrays and high-bandwidth HBM to execute bf16 matrix contractions at high utilization."
    neg_docs = [
        "Classic tiramisu combines espresso-soaked ladyfingers with mascarpone cream.",
        "Pacific gray whales migrate thousands of miles along the North American coast.",
        "Sourdough bread fermentation relies on wild yeast and lactobacilli.",
    ]

    def post_rerank(q: str, docs: list[str]) -> tuple[float, list[dict]]:
        payload = {"model": model_name, "query": q, "documents": docs}
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/v1/rerank",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=180) as r:
            obj = json.loads(r.read().decode("utf-8"))
        return (time.perf_counter() - t0) * 1000.0, obj["results"]

    _, q_res = post_rerank(query, [neg_docs[0], neg_docs[1], pos_doc, neg_docs[2]])
    q_res_sorted = sorted(q_res, key=lambda x: x["relevance_score"], reverse=True)
    top1_idx = q_res_sorted[0]["index"]
    top1_score = round(q_res_sorted[0]["relevance_score"], 4)
    bot_score = round(q_res_sorted[-1]["relevance_score"], 4)
    valid = top1_idx == 2 and top1_score > bot_score + 0.2

    records: list[dict] = []
    for n_cand, cs in [(10, [1, 4, 8]), (50, [1, 4]), (100, [1, 4])]:
        docs = [pos_doc] + [neg_docs[i % len(neg_docs)] for i in range(n_cand - 1)]
        for c in cs:
            n_q = max(4, c * 4) if n_cand == 10 else max(4, c * 2)
            post_rerank(query, docs)
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=c) as ex:
                res = list(ex.map(lambda _: post_rerank(query, docs)[0], range(n_q)))
            wall = time.perf_counter() - t0
            res.sort()
            rec = {
                "category": "reranker",
                "model": model_name,
                "engine": "vllm-torchtpu-0.33.0",
                "chip": "v7x",
                "tp": tp,
                "candidates_per_query": n_cand,
                "c": c,
                "n_queries": n_q,
                "wall_s": round(wall, 3),
                "p50_ms": round(res[len(res) // 2], 2),
                "p95_ms": round(res[int(0.95 * (len(res) - 1))], 2),
                "queries_per_sec": round(n_q / wall, 2),
                "pairs_per_sec": round((n_q * n_cand) / wall, 2),
                "top1_score": top1_score,
                "bottom_score": bot_score,
                "valid_ranking": valid,
                "status": "ok",
            }
            print(json.dumps(rec), flush=True)
            records.append(rec)
            if out_path:
                with out_path.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
    return records


def run_vllm_embed(base_url: str, model_name: str, tp: int, out_path: Path | None) -> list[dict]:
    def post_emb(inputs: list[str]) -> tuple[float, list[list[float]], int]:
        payload = {"model": model_name, "input": inputs}
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/v1/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=180) as r:
            obj = json.loads(r.read().decode("utf-8"))
        dt = (time.perf_counter() - t0) * 1000.0
        vecs = [d["embedding"] for d in obj["data"]]
        toks = (obj.get("usage") or {}).get("total_tokens", 0)
        return dt, vecs, toks

    _, vecs, _ = post_emb([
        "Instruct: Retrieve relevant passages.\nQuery: How do TPUs accelerate matrix multiplication?",
        "TPUs use systolic array matrix units (MXUs) and high-bandwidth HBM to accelerate dense tensor operations.",
        "A traditional Italian tiramisu recipe layers espresso-soaked ladyfingers with mascarpone cream.",
    ])
    sim_pos = round(cos_sim(vecs[0], vecs[1]), 4)
    sim_neg = round(cos_sim(vecs[0], vecs[2]), 4)
    dim = len(vecs[0])
    valid = sim_pos > sim_neg + 0.15

    short_doc = "Cloud TPU v7x compiles PyTorch computational graphs via TorchTPU for enterprise serving."
    med_doc = " ".join([short_doc] * 8)
    records: list[dict] = []
    for doc_name, doc_txt in [("short_25tok", short_doc), ("paragraph_200tok", med_doc)]:
        for bs in [1, 8, 32]:
            for c in [1, 4, 8]:
                if bs == 32 and c == 8:
                    continue
                n_reqs = max(8, c * 4)
                batch = [doc_txt] * bs
                post_emb(batch)
                t0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=c) as ex:
                    res = list(ex.map(lambda _: post_emb(batch), range(n_reqs)))
                wall = time.perf_counter() - t0
                lats = sorted(r[0] for r in res)
                tot_toks = sum(r[2] for r in res)
                rec = {
                    "category": "embedding",
                    "model": model_name,
                    "engine": "vllm-torchtpu-0.33.0",
                    "chip": "v7x",
                    "tp": tp,
                    "doc_profile": doc_name,
                    "batch_size": bs,
                    "c": c,
                    "n_reqs": n_reqs,
                    "wall_s": round(wall, 3),
                    "p50_ms": round(lats[len(lats) // 2], 2),
                    "p95_ms": round(lats[int(0.95 * (len(lats) - 1))], 2),
                    "docs_per_sec": round((n_reqs * bs) / wall, 2),
                    "tokens_per_sec": round(tot_toks / wall, 1),
                    "dim": dim,
                    "sim_pos": sim_pos,
                    "sim_neg": sim_neg,
                    "valid_semantic": valid,
                    "status": "ok",
                }
                print(json.dumps(rec), flush=True)
                records.append(rec)
                if out_path:
                    with out_path.open("a") as f:
                        f.write(json.dumps(rec) + "\n")
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark retrieval embeddings and rerankers on TPU v7x.")
    ap.add_argument("--mode", choices=["native", "vllm-embed", "vllm-rerank"], required=True)
    ap.add_argument("--model", required=True, help="Hugging Face model ID or local path")
    ap.add_argument("--model-name", default=None, help="Served/display model name")
    ap.add_argument("--task", choices=["embedding", "reranker"], default="embedding")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--seq-lens", default="64,256,512")
    ap.add_argument("--batch-sizes", default="1,8,32,64")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_name = args.model_name or args.model
    out_path = Path(args.out) if args.out else None
    if args.mode == "native":
        run_native_compiled(
            args.model,
            model_name,
            args.task,
            [int(x) for x in args.seq_lens.split(",")],
            [int(x) for x in args.batch_sizes.split(",")],
            args.iters,
            out_path,
        )
    elif args.mode == "vllm-embed":
        run_vllm_embed(args.base_url, model_name, args.tp, out_path)
    else:
        run_vllm_rerank(args.base_url, model_name, args.tp, out_path)


if __name__ == "__main__":
    main()
