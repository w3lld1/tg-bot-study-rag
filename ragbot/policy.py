from dataclasses import dataclass
from typing import Any, Dict


_ALLOWED_VARIANTS = {"control", "ab_retrieval_v1"}
_ALLOWED_INTENTS = {
    "default",
    "summary",
    "compare",
    "citation_only",
    "numbers_and_dates",
    "definition",
    "procedure",
    "requirements",
}


@dataclass(frozen=True)
class RetrievalRule:
    cov_threshold: float
    force_multiquery: bool


# Phase B: policy table (intent-aware) + A/B variants.
# NOTE: `control` preserves current behavior by default.
_POLICY_TABLE: Dict[str, Dict[str, Dict[str, Any]]] = {
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
    # AB hypothesis rationale:
    # - summary/definition/procedure/requirements are often semantically wide in mixed PDFs,
    #   so we slightly raise cov threshold to trigger multiquery earlier.
    # - compare uses smaller uplift to avoid over-expansion noise.
    # - numbers_and_dates remains unchanged (already force_multiquery=True).
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


def _strict_mode(settings: Any) -> bool:
    return bool(getattr(settings, "policy_strict", False))


def _normalize_variant(settings: Any) -> str:
    return str(getattr(settings, "policy_variant", "control") or "control").strip().lower()


def get_policy_variant(settings: Any) -> str:
    variant = _normalize_variant(settings)
    if variant in _ALLOWED_VARIANTS:
        return variant
    if _strict_mode(settings):
        raise ValueError(f"Unknown policy_variant='{variant}'. Allowed: {sorted(_ALLOWED_VARIANTS)}")
    return "control"


def _parse_rule(raw: Dict[str, Any], *, variant: str, intent: str) -> RetrievalRule:
    if not isinstance(raw, dict):
        raise ValueError(f"Policy rule must be object, got {type(raw)} for {variant}/{intent}")
    if "cov_threshold" not in raw or "force_multiquery" not in raw:
        raise ValueError(f"Policy rule missing keys for {variant}/{intent}: {raw}")

    cov = float(raw["cov_threshold"])
    if cov < 0.0 or cov > 1.0:
        raise ValueError(f"cov_threshold out of range [0,1] for {variant}/{intent}: {cov}")

    return RetrievalRule(cov_threshold=cov, force_multiquery=bool(raw["force_multiquery"]))


def _validate_policy_table() -> None:
    for variant, table in _POLICY_TABLE.items():
        if variant not in _ALLOWED_VARIANTS:
            raise ValueError(f"Unknown variant in _POLICY_TABLE: {variant}")
        if "default" not in table:
            raise ValueError(f"Variant '{variant}' must contain 'default' rule")

        for intent, raw_rule in table.items():
            if intent not in _ALLOWED_INTENTS:
                raise ValueError(f"Unknown intent '{intent}' in variant '{variant}'")
            _parse_rule(raw_rule, variant=variant, intent=intent)


_validate_policy_table()


def get_retrieval_policy(settings: Any, intent: str) -> Dict[str, Any]:
    variant = get_policy_variant(settings)
    table = _POLICY_TABLE[variant]

    if intent not in _ALLOWED_INTENTS:
        if _strict_mode(settings):
            raise ValueError(f"Unknown intent='{intent}'. Allowed: {sorted(_ALLOWED_INTENTS)}")
        intent = "default"

    fallback = table["default"]
    raw_rule = table.get(intent, fallback)
    rule = _parse_rule(raw_rule, variant=variant, intent=intent)

    return {
        "variant": variant,
        "intent": intent,
        "cov_threshold": rule.cov_threshold,
        "force_multiquery": rule.force_multiquery,
        "rule_id": f"{variant}:{intent}",
    }


def should_trigger_multiquery(*, multiquery_enabled: bool, force_multiquery: bool, coverage: float, cov_threshold: float) -> bool:
    return bool(multiquery_enabled and (force_multiquery or coverage < cov_threshold))


def get_second_pass_overrides(settings: Any, intent: str) -> Dict[str, Any]:
    variant = get_policy_variant(settings)
    if intent not in _ALLOWED_INTENTS and _strict_mode(settings):
        raise ValueError(f"Unknown intent='{intent}' for second pass")

    mult = int(getattr(settings, "second_pass_k_multiplier", 5))

    # AB hypothesis: for wider semantic intents, one extra multiplier step on second pass.
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
