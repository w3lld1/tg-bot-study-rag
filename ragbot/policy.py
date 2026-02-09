from typing import Any, Dict


_ALLOWED_VARIANTS = {"control", "ab_retrieval_v1"}


# Phase B: policy table (intent-aware) + A/B variants.
# NOTE: `control` preserves current behavior by default.
_POLICY_TABLE: Dict[str, Dict[str, Any]] = {
    "control": {
        "default": {"cov_threshold": 0.50, "force_multiquery": False},
        "summary": {"cov_threshold": 0.50, "force_multiquery": False},
        "compare": {"cov_threshold": 0.50, "force_multiquery": False},
        "citation_only": {"cov_threshold": 0.50, "force_multiquery": False},
        "numbers_and_dates": {"cov_threshold": 0.50, "force_multiquery": True},
        "definition": {"cov_threshold": 0.50, "force_multiquery": False},
        "procedure": {"cov_threshold": 0.50, "force_multiquery": False},
        "requirements": {"cov_threshold": 0.50, "force_multiquery": False},
    },
    # A/B variant for future tests: slightly more eager multiquery on non-numeric intents.
    "ab_retrieval_v1": {
        "default": {"cov_threshold": 0.58, "force_multiquery": False},
        "summary": {"cov_threshold": 0.60, "force_multiquery": False},
        "compare": {"cov_threshold": 0.56, "force_multiquery": False},
        "citation_only": {"cov_threshold": 0.50, "force_multiquery": False},
        "numbers_and_dates": {"cov_threshold": 0.50, "force_multiquery": True},
        "definition": {"cov_threshold": 0.60, "force_multiquery": False},
        "procedure": {"cov_threshold": 0.56, "force_multiquery": False},
        "requirements": {"cov_threshold": 0.56, "force_multiquery": False},
    },
}


def get_policy_variant(settings: Any) -> str:
    variant = str(getattr(settings, "policy_variant", "control") or "control").strip().lower()
    return variant if variant in _ALLOWED_VARIANTS else "control"


def get_retrieval_policy(settings: Any, intent: str) -> Dict[str, Any]:
    variant = get_policy_variant(settings)
    table = _POLICY_TABLE.get(variant, _POLICY_TABLE["control"])
    fallback = table.get("default", _POLICY_TABLE["control"]["default"])
    policy = dict(table.get(intent, fallback))
    policy["variant"] = variant
    policy["intent"] = intent
    return policy


def get_second_pass_overrides(settings: Any, intent: str) -> Dict[str, Any]:
    variant = get_policy_variant(settings)
    mult = int(getattr(settings, "second_pass_k_multiplier", 5))

    # Keep control behavior as-is.
    if variant == "ab_retrieval_v1" and intent in {"summary", "compare", "definition", "default"}:
        mult = max(1, mult + 1)

    return {
        "multiquery_n": int(getattr(settings, "second_pass_multiquery_n", 6)),
        "multiquery_k_per_query": max(int(getattr(settings, "multiquery_k_per_query", 12)) * mult, int(getattr(settings, "multiquery_k_per_query", 12)) + 30),
        "bm25_k": max(int(getattr(settings, "bm25_k", 14)) * mult, int(getattr(settings, "bm25_k", 14)) + 30),
        "faiss_k": max(int(getattr(settings, "faiss_k", 14)) * mult, int(getattr(settings, "faiss_k", 14)) + 30),
        "numbers_k_multiplier": max(int(getattr(settings, "numbers_k_multiplier", 3)), 1),
        "numbers_neighbors_window": int(getattr(settings, "numbers_neighbors_window", 2)),
        "numbers_disable_diversify": bool(getattr(settings, "numbers_disable_diversify", True)),
        "rerank_keep": max(int(getattr(settings, "rerank_keep", 10)), 14),
        "rerank_pool": int(getattr(settings, "rerank_pool", 18)),
        "policy_variant": variant,
    }
