# TorchTPU Retrieval, Embeddings & Reranking Serving on Cloud TPU v7x

Production-grade dense embedding and cross-encoder reranking benchmarks on **Google Cloud TPU v7x** using PyTorch with TorchTPU (`torch.compile(backend="tpu")`) and OpenAI-compatible `vllm serve --runner pooling` (`vllm-torchtpu v0.33.0`). On a single TPU v7x core, compiled `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3` reach **2,460 docs/s** and **2,390 pairs/s** (`1.28–1.33 ms` `B=1` latency, **305K tokens/s** at `seq=512`), while live `vllm` pooling servers on 1 TPU v7x chip (`TP=2`) sustain **977.3 docs/s** for `Qwen/Qwen3-Embedding-8B` and **723.7 pairs/s** for `Qwen/Qwen3-Reranker-8B` with verified semantic separation and top-1 ranking accuracy.

---

## Models, Frameworks & Hardware

| Model | Role | Architecture | Params / Precision | Output / Verification | License |
|---|---|---|---|---|---|
| [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3) | Multilingual dense embedder | Bidirectional XLM-RoBERTa encoder | 568M / `bf16` | 1024-dim (`0.728` pos vs `0.302` neg cosine) | MIT |
| [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Multilingual cross-encoder | Bidirectional XLM-RoBERTa classifier | 568M / `bf16` | Scalar logit (`+3.98` pos vs `-11.06` neg) | Apache-2.0 |
| [`Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B) | LLM dense embedder | Causal Qwen3 decoder (last-token pool) | 8.0B / `bf16` | 4096-dim (`0.875` pos vs `0.174` neg cosine) | Apache-2.0 |
| [`Qwen/Qwen3-Reranker-8B`](https://huggingface.co/Qwen/Qwen3-Reranker-8B) | LLM cross-encoder reranker | Causal Qwen3 yes/no classifier | 8.0B / `bf16` | Prob score (`0.922` top-1 vs `0.139` bottom) | Apache-2.0 |

* **Hardware:** Google Cloud TPU v7x (2 TensorCores per chip, 192 GB HBM per chip). Native `torch.compile(backend="tpu")` runs on **1 single TPU v7x core** (half a chip); live `vllm --runner pooling` HTTP servers run at **`TP=2` (1 TPU v7x chip)**.
* **Software stack:** PyTorch + TorchTPU (`torch_tpu`), `vllm-torchtpu v0.33.0`, Hugging Face `transformers`. Also audited on `sglang-torchtpu v0.2.0` (`--is-embedding` on `Qwen3-Embedding-8B` and `BAAI/bge-m3` aborts in XLA at `windowing_util.cc:318`; full trace in [APPENDIX.md](APPENDIX.md)).

---

## Main Results

### 1. Native `torch.compile(backend="tpu")` — Single TPU v7x Core (`bf16`)

Measured warm steady-state execution after graph compilation on 1 TPU v7x core (`results/v7x_native_compiled_embeddings_rerankers.jsonl`):

| Model | Seq Len | `B=1` p50 Latency | Peak Throughput (`items/s`) | Peak Token Rate (`tok/s`) | Best Batch |
|---|---:|---:|---:|---:|---:|
| **`BAAI/bge-m3`** | 64 | **1.33 ms** | **2,460.1 docs/s** | 157,446 | `B=8` |
| **`BAAI/bge-m3`** | 256 | 1.47 ms | 1,043.0 docs/s | 267,015 | `B=64` |
| **`BAAI/bge-m3`** | 512 | 1.78 ms | 596.3 docs/s | **305,327** | `B=64` |
| **`BAAI/bge-reranker-v2-m3`** | 64 | **1.28 ms** | **2,389.6 pairs/s** | 152,934 | `B=8` |
| **`BAAI/bge-reranker-v2-m3`** | 256 | 1.39 ms | 1,039.9 pairs/s | 266,209 | `B=64` |
| **`BAAI/bge-reranker-v2-m3`** | 512 | 1.73 ms | 595.9 pairs/s | **305,073** | `B=64` |
| **`Qwen/Qwen3-Embedding-8B`** | 64 | **7.57 ms** | **453.8 docs/s** | 29,042 | `B=32` |
| **`Qwen/Qwen3-Embedding-8B`** | 256 | 8.17 ms | 168.8 docs/s | 43,213 | `B=32` |
| **`Qwen/Qwen3-Embedding-8B`** | 512 | 13.29 ms | 89.6 docs/s | **45,864** | `B=32` |

### 2. Live OpenAI-Compatible `vllm serve --runner pooling` — 1 TPU v7x Chip (`TP=2`)

End-to-end HTTP JSON serving (`/v1/embeddings` and `/v1/rerank`) on `vllm-torchtpu v0.33.0`:

#### `Qwen/Qwen3-Embedding-8B` (`/v1/embeddings`, `TP=2`)

| Input Profile | Batch / Request | Concurrency | p50 Latency | p95 Latency | Throughput (`docs/s`) | Token Rate (`tok/s`) |
|---|---:|---:|---:|---:|---:|---:|
| Short query (~27 tok) | 1 | 1 | **8.69 ms** | 9.19 ms | 110.7 | 2,989 |
| Short query (~27 tok) | 8 | 8 | 74.27 ms | 129.27 ms | 753.1 | 20,333 |
| Short query (~27 tok) | 32 | 4 | 108.93 ms | 171.45 ms | **977.3** | 26,387 |
| RAG paragraph (~216 tok) | 1 | 1 | **9.99 ms** | 10.27 ms | 79.5 | 17,182 |
| RAG paragraph (~216 tok) | 8 | 8 | 231.24 ms | 237.66 ms | **272.6** | **58,874** |

#### `Qwen/Qwen3-Reranker-8B` (`/v1/rerank`, `TP=2`)

| Candidates / Query | Concurrency | p50 Latency | p95 Latency | Queries / Sec | Pairs / Sec | Top-1 Verified |
|---:|---:|---:|---:|---:|---:|:---:|
| 10 | 1 | **64.85 ms** | 102.14 ms | 15.6 | 156.2 | Yes (`0.922` vs `0.139`) |
| 10 | 8 | 126.08 ms | 204.88 ms | **53.5** | 535.0 | Yes |
| 50 | 1 | 116.70 ms | 155.92 ms | 8.6 | 430.2 | Yes |
| 50 | 4 | 268.18 ms | 378.02 ms | 13.3 | 663.0 | Yes |
| 100 | 1 | 193.80 ms | 219.17 ms | 5.2 | 521.6 | Yes |
| 100 | 4 | 524.24 ms | 562.88 ms | 7.2 | **723.7** | Yes |

* **Serving architecture takeaway:** On `vllm-torchtpu v0.33.0`, causal decoder pooling models (`Qwen3-Embedding-8B`, `Qwen3-Reranker-8B`) run directly via `vllm serve --runner pooling` with OpenAI-compatible HTTP APIs, while bidirectional XLM-RoBERTa encoders (`BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`) achieve sub-2 ms single-core execution via native `torch.compile(backend="tpu")` (details in [APPENDIX.md](APPENDIX.md)).

---

## How to Reproduce

### Native `torch.compile(backend="tpu")` (1 TPU v7x Core)

```bash
# BAAI/bge-m3 dense embeddings
python3 scripts/benchmark_retrieval.py \
  --mode native --task embedding \
  --model BAAI/bge-m3 --seq-lens 64,256,512 --batch-sizes 1,8,32,64

# BAAI/bge-reranker-v2-m3 cross-encoder reranking
python3 scripts/benchmark_retrieval.py \
  --mode native --task reranker \
  --model BAAI/bge-reranker-v2-m3 --seq-lens 64,256,512 --batch-sizes 1,8,32,64

# Qwen/Qwen3-Embedding-8B (1 core)
python3 scripts/benchmark_retrieval.py \
  --mode native --task embedding \
  --model Qwen/Qwen3-Embedding-8B --seq-lens 64,256,512 --batch-sizes 1,8,32
```

### Live `vllm serve --runner pooling` (`TP=2`, 1 TPU v7x Chip)

```bash
# Terminal 1: Start Qwen3-Embedding-8B HTTP server
vllm serve Qwen/Qwen3-Embedding-8B \
  --runner pooling --tensor-parallel-size 2 --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --trust-remote-code --port 8000

# Terminal 2: Benchmark /v1/embeddings
python3 scripts/benchmark_retrieval.py \
  --mode vllm-embed --model Qwen/Qwen3-Embedding-8B --base-url http://127.0.0.1:8000 --tp 2
```

```bash
# Terminal 1: Start Qwen3-Reranker-8B HTTP server (yes/no classifier head)
vllm serve Qwen/Qwen3-Reranker-8B \
  --runner pooling --tensor-parallel-size 2 --max-model-len 8192 \
  --gpu-memory-utilization 0.85 --trust-remote-code --port 8001 \
  --hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}'

# Terminal 2: Benchmark /v1/rerank
python3 scripts/benchmark_retrieval.py \
  --mode vllm-rerank --model Qwen/Qwen3-Reranker-8B --base-url http://127.0.0.1:8001 --tp 2
```

---

## Full Data & Engineering Notes

* [APPENDIX.md](APPENDIX.md) — Complete batch/sequence sweeps, `vllm --runner pooling` bidirectional attention diagnostics (`XLM-RoBERTa`), `sglang-torchtpu v0.2.0` `--is-embedding` XLA abort traces, deployment notes, and cold-compile timings.
* [WORKING_LOG.md](WORKING_LOG.md) — Chronological engineering bring-up and dual-engine audit log.
* [`results/`](results/) — Raw verified JSONL measurements backing every table above.
