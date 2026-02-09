from ragbot.policy import (
    get_policy_variant,
    get_retrieval_policy,
    get_second_pass_overrides,
    should_trigger_multiquery,
)


class _S:
    policy_variant = "control"
    policy_strict = False
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


def test_policy_variant_fallback_to_control_non_strict():
    s = _S()
    s.policy_variant = "unknown"
    s.policy_strict = False
    assert get_policy_variant(s) == "control"


def test_policy_variant_strict_raises_on_unknown():
    s = _S()
    s.policy_variant = "unknown"
    s.policy_strict = True
    try:
        get_policy_variant(s)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_retrieval_policy_decision_fields_present():
    s = _S()
    p = get_retrieval_policy(s, "summary")
    assert p["variant"] == "control"
    assert p["rule_id"] == "control:summary"
    assert 0.0 <= p["cov_threshold"] <= 1.0


def test_multiquery_trigger_behavioral_contract():
    # below threshold -> trigger
    assert should_trigger_multiquery(
        multiquery_enabled=True,
        force_multiquery=False,
        coverage=0.49,
        cov_threshold=0.50,
    )
    # above threshold -> no trigger
    assert not should_trigger_multiquery(
        multiquery_enabled=True,
        force_multiquery=False,
        coverage=0.51,
        cov_threshold=0.50,
    )
    # force flag dominates
    assert should_trigger_multiquery(
        multiquery_enabled=True,
        force_multiquery=True,
        coverage=0.99,
        cov_threshold=0.10,
    )
    # disabled dominates everything
    assert not should_trigger_multiquery(
        multiquery_enabled=False,
        force_multiquery=True,
        coverage=0.0,
        cov_threshold=1.0,
    )


def test_ab_variant_changes_second_pass_multiplier_for_target_intents():
    s = _S()
    s.policy_variant = "ab_retrieval_v1"
    s.policy_strict = True

    # default intent is uplifted by +1 multiplier in ab variant
    o = get_second_pass_overrides(s, "default")
    # base bm25_k=14, mult=6 -> 84
    assert o["bm25_k"] >= 84
    assert o["policy_variant"] == "ab_retrieval_v1"


def test_control_variant_second_pass_baseline_multiplier():
    s = _S()
    s.policy_variant = "control"
    s.policy_strict = True

    o = get_second_pass_overrides(s, "default")
    # base bm25_k=14, mult=5 -> 70
    assert o["bm25_k"] >= 70
    assert o["bm25_k"] < 84
