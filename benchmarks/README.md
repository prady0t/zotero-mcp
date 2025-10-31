## Embedding Benchmark Utilities

This folder contains assets for evaluating embedding models against real Zotero-style corpora.

- `queries/semantic_retrieval_queries.json` – curated natural-language queries with graded relevance, used by `mteb_integration_benchmark.py`.
- `results/` – default location for benchmark outputs (`summary.json`, `summary.csv`, `query_breakdown.json`, and per-query analysis dumps).

### Running the benchmark

```bash
pixi run python mteb_integration_benchmark.py \
  --models Qwen/Qwen3-Embedding-0.6B \
  --query-file benchmarks/queries/semantic_retrieval_queries.json \
  --output-dir benchmarks/results/latest_run
```

By default the script evaluates the ChromaDB baseline and any additional models you list, across both abstract and full-text retrieval modes, emits a comparative scoreboard, and writes per-query diagnostics (top-K documents) when `--analysis-top-k` is greater than zero (default is 5). Set `--no-default-model` to skip the baseline or `--modes abstract` to disable full-text chunk evaluation.

`query_breakdown.json` summarises hit rates by category and difficulty for each model/mode pair, making it easy to spot where stronger embeddings actually help.
