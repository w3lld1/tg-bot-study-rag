from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_EVAL = (ROOT / "eval" / "run_eval.py").read_text(encoding="utf-8")
RUN_BENCH = (ROOT / "eval" / "benchmark" / "run_benchmark.py").read_text(encoding="utf-8")


def test_run_eval_has_reusable_index_flags():
    assert "--index-dir" in RUN_EVAL
    assert "--no-reuse-index" in RUN_EVAL
    assert "def _index_ready(" in RUN_EVAL
    assert "def _maybe_ingest(" in RUN_EVAL


def test_run_benchmark_has_index_cache_flags():
    assert "--index-cache-dir" in RUN_BENCH
    assert "--no-reuse-index" in RUN_BENCH
    assert "index_dir = index_cache_dir" in RUN_BENCH
