# Working Log: Retrieval, Embeddings & Reranking Bring-Up on TPU v7x

Chronological engineering log of license verification, native `torch.compile(backend="tpu")` benchmarking, `vllm-torchtpu v0.33.0` `--runner pooling` bring-up, and `sglang-torchtpu v0.2.0` dual-engine auditing on Google Cloud TPU v7x.

## 1. License Verification

- Confirmed open-source licenses before downloading weights:
  - `BAAI/bge-m3` (MIT)
  - `BAAI/bge-reranker-v2-m3` (Apache-2.0)
  - `Qwen/Qwen3-Embedding-8B` (Apache-2.0)
  - `Qwen/Qwen3-Reranker-8B` (Apache-2.0)

## 2. Native `torch.compile(backend="tpu")` Benchmarks (Single Core, `bf16`)

- **`BAAI/bge-m3`** (`568M`): Compiled cleanly on `tpu:0`, achieving `1.33 ms` (`B=1, seq=64`), `2,460.1 docs/s` (`B=8, seq=64`), and `305,327 tok/s` (`B=64, seq=512`), passing semantic separation (`0.7284` positive vs `0.3017` negative cosine).
- **`BAAI/bge-reranker-v2-m3`** (`568M`): Compiled cleanly on `tpu:0`, achieving `1.28 ms` (`B=1, seq=64`), `2,389.6 pairs/s` (`B=8, seq=64`), and `305,073 tok/s` (`B=64, seq=512`), passing ranking gate (`+3.9844` positive vs `-11.0625` negative logit).
- **`Qwen/Qwen3-Embedding-8B`** (`8.0B`): Fit on 1 single core (`15.2 GB` in `bf16`), achieving `7.57 ms` (`B=1, seq=64`), `453.8 docs/s` (`B=32, seq=64`), and `45,864 tok/s` (`B=32, seq=512`) with strong semantic separation (`0.8751` vs `0.1738`).

## 3. Live `vllm serve --runner pooling` (`vllm-torchtpu v0.33.0`, `TP=2`)

- **Bidirectional encoder finding (`BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`):** `vllm serve --runner pooling` raised `NotImplementedError` inside `vllm/v1/attention/backend.py:94` (`get_builder_cls`) because `vllm-torchtpu v0.33.0` implements causal Pallas FlashAttention rather than bidirectional encoder attention.
- **`Qwen/Qwen3-Embedding-8B` (`/v1/embeddings`):** Booted cleanly at `TP=2` in `136.0 s`, sustaining `8.69 ms` (`B=1, c=1`) up to **977.3 docs/s** (`B=32, c=4`) and **58,874 tok/s** (`B=8, c=8`, RAG paragraphs).
- **`Qwen/Qwen3-Reranker-8B` (`/v1/rerank`):** Configured with `--hf-overrides '{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}'`. Sustained **723.7 pairs/s** (`100` candidates/query, `c=4`) with verified top-1 relevance (`0.9224` vs `0.1385`).

## 4. Dual-Engine Audit (`sglang-torchtpu v0.2.0`, `--is-embedding`)

- Tested `Qwen/Qwen3-Embedding-8B` (`TP=2`) and `BAAI/bge-m3` (`TP=1`) with `sglang.launch_server --device tpu --is-embedding`.
- `Qwen3-Embedding-8B` loaded weights in `105.0 s` and passed `/health`, then crashed on the first `/v1/embeddings` request with a fatal XLA abort (`exit -6`, `windowing_util.cc:318: Check failed: desc->window_bounds.size() == rank (0 vs. 1)`).
- `BAAI/bge-m3` loaded weights in `63.0 s` and hit the identical fatal XLA abort (`windowing_util.cc:318`) during warmup prefill.
- Recorded raw diagnostics in `results/v7x_sglang_dual_engine_audit.jsonl` and `APPENDIX.md`.
