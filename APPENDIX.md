# Appendix: Detailed Sweeps & Runtime Diagnostics

This appendix records full sequence-length and batch-size progressions, cold compilation overhead, and runtime compatibility findings for retrieval embedding and cross-encoder reranking workloads on Google Cloud TPU v7x.

---

## 1. Runtime Compatibility: `vllm --runner pooling` vs Native `torch.compile(backend="tpu")`

### Bidirectional Encoder Models (`BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`)

* **Observation in `vllm-torchtpu v0.33.0`:** Launching `vllm serve BAAI/bge-m3 --runner pooling` or `BAAI/bge-reranker-v2-m3 --runner pooling` fails during worker initialization inside `vllm/v1/attention/backend.py:94` (`get_builder_cls -> NotImplementedError`) because the `vllm-torchtpu` attention backend implements causal Pallas FlashAttention rather than bidirectional encoder attention.
* **Recommended TPU serving path:** Execute bidirectional XLM-RoBERTa encoders directly via `torch.compile(backend="tpu")` in `bf16` (wrapped in a lightweight FastAPI/gRPC microservice with static shape bucketing across `(batch_size, seq_len)`). Each model occupies under 1.2 GB of HBM and runs independently on a single TPU v7x core, enabling 2 independent replicas per chip (8 replicas per 4-chip host) with linear throughput scaling.

### Causal Decoder Pooling Models (`Qwen/Qwen3-Embedding-8B`, `Qwen/Qwen3-Reranker-8B`)

* **Live `vllm --runner pooling` support:** Both models use the causal Qwen3 backbone (`PallasAttentionBackendImpl` across 36 layers) and boot cleanly on `vllm-torchtpu v0.33.0` at `TP=2` (1 TPU v7x chip):
  * `Qwen/Qwen3-Embedding-8B`: booted in `136.0 s` (warm local/cached weights) and served `/v1/embeddings` at `8.69 ms` (`B=1, c=1`) up to **977.3 docs/s** (`B=32, c=4`, short queries) and **58,874 tok/s** (`B=8, c=8`, 200-token RAG chunks).
  * `Qwen/Qwen3-Reranker-8B`: requires `--hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}'` to map the pretrained `"yes"`/`"no"` LM head rows into a single relevance logit. Served `/v1/rerank` up to **723.7 pairs/s** (`100` candidates/query, `c=4`) with verified top-1 accuracy (`0.9224` relevant vs `0.1385` bottom irrelevant).
* **Single-core vs `TP=2` comparison for `Qwen3-Embedding-8B`:**
  * On **1 single core** (half a chip), native `torch.compile(backend="tpu")` sustains `453.8 docs/s` (`seq=64, B=32`) and `45,864 tok/s` (`seq=512, B=32`).
  * On **1 full chip (`TP=2`)**, live HTTP `vllm --runner pooling` sustains `977.3 docs/s` (short ~27-tok inputs) and `58,874 tok/s` (216-tok RAG paragraphs) including JSON serialization, tokenization, and continuous batching.
  * Because `Qwen3-Embedding-8B` weighs ~15.2 GB in `bf16` and each TPU v7x core has 96 GB HBM, running two independent `TP=1` replicas per chip can double aggregate throughput relative to `TP=2` when single-request latency under 10 ms is not strictly required.
* **FUSE storage note:** On FUSE-mounted checkpoint directories, directory metadata caching can occasionally hide newly written `.safetensors` shards from glob discovery (`vllm`'s `DefaultModelLoader`) while direct file paths succeed. Ensuring checkpoint directories are materialized before server startup avoids false `Cannot find any model weights` errors.

---

## 2. Full Native `torch.compile(backend="tpu")` Sweep (1 TPU v7x Core, `bf16`)

### `BAAI/bge-m3` (`dim = 1024`, `sim_pos = 0.7284`, `sim_neg = 0.3017`)

| Seq Len | Batch | Cold Compile (ms) | p50 (ms) | p95 (ms) | Docs / s | Tokens / s |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 5,390.8 | 1.33 | 1.37 | 753.5 | 48,226 |
| 64 | 8 | 14,723.6 | 3.25 | 3.32 | **2,460.1** | 157,446 |
| 64 | 32 | 12,717.2 | 16.07 | 16.38 | 1,991.2 | 127,439 |
| 64 | 64 | 14,534.0 | 31.69 | 32.05 | 2,019.5 | 129,247 |
| 256 | 1 | 5,968.2 | 1.47 | 1.74 | 681.3 | 174,415 |
| 256 | 8 | 13,352.9 | 8.22 | 8.28 | 973.8 | 249,291 |
| 256 | 32 | 17,301.3 | 33.64 | 33.89 | 951.3 | 243,536 |
| 256 | 64 | 20,801.7 | 61.36 | 61.72 | **1,043.0** | 267,015 |
| 512 | 1 | 6,974.3 | 1.78 | 1.80 | 562.3 | 287,896 |
| 512 | 8 | 15,341.8 | 15.98 | 16.52 | 500.6 | 256,294 |
| 512 | 32 | 20,322.8 | 55.49 | 55.77 | 576.7 | 295,287 |
| 512 | 64 | 23,897.0 | 107.32 | 107.64 | 596.3 | **305,327** |

### `BAAI/bge-reranker-v2-m3` (`score_pos = +3.9844`, `score_neg = -11.0625`, `valid_ranking = true`)

| Seq Len | Batch | Cold Compile (ms) | p50 (ms) | p95 (ms) | Pairs / s | Tokens / s |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 5,532.1 | 1.28 | 1.33 | 782.7 | 50,094 |
| 64 | 8 | 14,687.0 | 3.35 | 3.62 | **2,389.6** | 152,934 |
| 64 | 32 | 12,621.9 | 15.61 | 16.36 | 2,049.8 | 131,190 |
| 64 | 64 | 14,402.9 | 31.43 | 31.59 | 2,036.4 | 130,330 |
| 256 | 1 | 5,872.2 | 1.39 | 1.45 | 717.8 | 183,744 |
| 256 | 8 | 12,755.2 | 8.20 | 8.28 | 975.6 | 249,747 |
| 256 | 32 | 16,951.5 | 33.86 | 34.10 | 945.1 | 241,952 |
| 256 | 64 | 20,316.8 | 61.55 | 62.44 | **1,039.9** | 266,209 |
| 512 | 1 | 6,852.6 | 1.73 | 2.05 | 579.1 | 296,520 |
| 512 | 8 | 15,220.6 | 16.14 | 16.72 | 495.7 | 253,810 |
| 512 | 32 | 20,101.1 | 55.63 | 55.99 | 575.3 | 294,534 |
| 512 | 64 | 23,864.5 | 107.41 | 107.65 | 595.9 | **305,073** |

### `Qwen/Qwen3-Embedding-8B` (`dim = 4096`, `sim_pos = 0.8751`, `sim_neg = 0.1738`)

| Seq Len | Batch | Cold Compile (ms) | p50 (ms) | p95 (ms) | Docs / s | Tokens / s |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 1 | 12,490.2 | 7.57 | 7.60 | 132.1 | 8,454 |
| 64 | 8 | 13,828.3 | 18.79 | 19.25 | 425.8 | 27,248 |
| 64 | 32 | 15,902.9 | 70.52 | 71.61 | **453.8** | 29,042 |
| 256 | 1 | 13,423.3 | 8.17 | 8.54 | 122.4 | 31,328 |
| 256 | 8 | 15,955.2 | 53.83 | 54.14 | 148.6 | 38,046 |
| 256 | 32 | 16,936.3 | 189.57 | 189.85 | 168.8 | 43,212 |
| 512 | 1 | 14,339.5 | 13.29 | 13.79 | 75.2 | 38,516 |
| 512 | 8 | 16,492.0 | 92.32 | 92.51 | 86.7 | 44,368 |
| 512 | 32 | 19,317.0 | 357.23 | 357.62 | 89.6 | **45,864** |

---

## 3. Dual-Engine Audit (`sglang-torchtpu v0.2.0`)

Both `Qwen/Qwen3-Embedding-8B` (`TP=2`) and `BAAI/bge-m3` (`TP=1`) were tested live on TPU v7x using `sglang.launch_server --device tpu --is-embedding --dtype bfloat16` (`sglang-torchtpu v0.2.0`):

| Model | Engine | Config | Startup Progress | Outcome | Verbatim XLA Abort |
| :--- | :--- | :---: | :--- | :---: | :--- |
| `Qwen/Qwen3-Embedding-8B` | `sglang-torchtpu v0.2.0` | `TP=2`, `--is-embedding` | Loaded weights (`105.0 s`), `/health 200 OK` | Scheduler crashed (`exit -6`) on first `/v1/embeddings` request | `windowing_util.cc:318 Check failed: desc->window_bounds.size() == rank (0 vs. 1)` |
| `BAAI/bge-m3` | `sglang-torchtpu v0.2.0` | `TP=1`, `--is-embedding` | Loaded weights (`63.0 s`) | Scheduler crashed (`exit -6`) during warmup prefill | `windowing_util.cc:318 Check failed: desc->window_bounds.size() == rank (0 vs. 1)` |

### Verbatim XLA TPU Abort Trace (`sglang-torchtpu v0.2.0`, `--is-embedding`)

```text
F... windowing_util.cc:318] Check failed: desc->window_bounds.size() == rank (0 vs. 1) base_bounds: (2)
F... lowering_util.cc:4703] Check failed: !data->ProducesVmreg() && !data->ProducesVreg() Unsupported conversion from vmreg/vreg to U64
Fatal Python error: Aborted
Subprocess scheduler_0 crashed with exit code -6.
```

Because `sglang-torchtpu v0.2.0`'s embedding attention path hits this fatal XLA lowering check on TPU v7x, `vllm serve --runner pooling` (`vllm-torchtpu v0.33.0`) is the verified OpenAI-compatible HTTP server for `Qwen3-Embedding-8B` / `Qwen3-Reranker-8B`, and native `torch.compile(backend="tpu")` is the verified high-throughput path for `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3`.

---

## 4. Full Live `vllm --runner pooling` & Audit Result Files

Full raw records are committed in:
* [`results/v7x_vllm_pooling_qwen3_embedding_8b_tp2.jsonl`](results/v7x_vllm_pooling_qwen3_embedding_8b_tp2.jsonl)
* [`results/v7x_vllm_pooling_qwen3_reranker_8b_tp2.jsonl`](results/v7x_vllm_pooling_qwen3_reranker_8b_tp2.jsonl)
* [`results/v7x_vllm_pooling_diagnostics.jsonl`](results/v7x_vllm_pooling_diagnostics.jsonl)
* [`results/v7x_sglang_dual_engine_audit.jsonl`](results/v7x_sglang_dual_engine_audit.jsonl)

