from pathlib import Path

from ragbot.policy import get_policy_variant, get_retrieval_policy, get_second_pass_overrides


ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = (ROOT / "ragbot" / "agent.py").read_text(encoding="utf-8")


class _S:
    policy_variant = "control"
    second_pass_k_multiplier = 5
    second_pass_multiquery_n = 6
    multiquery_k_per_query = 12
    bm25_k = 14
    faiss_k = 14
    numbers_k_multiplier = 3
    numbers_neighbors_window = 2
    numbers_disable_diversify = True
    rerank_keep = 10
    rerank_pool = 18


def test_policy_variant_fallback_to_control():
    s = _S()
    s.policy_variant = "unknown"
    assert get_policy_variant(s) == "control"


def test_retrieval_policy_has_threshold_and_variant():
    s = _S()
    p = get_retrieval_policy(s, "summary")
    assert "cov_threshold" in p
    assert p["variant"] == "control"


def test_second_pass_overrides_control_preserve_shape():
    s = _S()
    o = get_second_pass_overrides(s, "default")
    assert o["bm25_k"] >= 44
    assert o["faiss_k"] >= 44
    assert o["policy_variant"] == "control"


def test_agent_uses_policy_helpers():
    assert "get_retrieval_policy" in AGENT_SRC
    assert "get_second_pass_overrides" in AGENT_SRC
