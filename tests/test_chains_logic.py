import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List

from ragbot.core_logic import INTENTS, detect_intent_fast, invoke_json_robust
from ragbot.text_utils import clamp_text, safe_page_range


CHAINS_PATH = Path(__file__).resolve().parents[1] / "ragbot" / "chains.py"


@dataclass
class DummyDoc:
    page_content: str
    metadata: dict


def load_chain_symbols(*names):
    src = CHAINS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)

    selected = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]

    mod = ast.Module(body=selected, type_ignores=[])
    code = compile(mod, filename=str(CHAINS_PATH), mode="exec")

    ns = {
        "INTENTS": INTENTS,
        "detect_intent_fast": detect_intent_fast,
        "invoke_json_robust": invoke_json_robust,
        "clamp_text": clamp_text,
        "safe_page_range": safe_page_range,
        "List": List,
        "Document": DummyDoc,
    }
    exec(code, ns, ns)
    return {name: ns[name] for name in names}


FUNCS = load_chain_symbols("get_intent_hybrid", "rerank_docs")


def test_get_intent_hybrid_uses_fast_path_when_confident():
    fn = FUNCS["get_intent_hybrid"]

    class Chain:
        def invoke(self, _):
            raise AssertionError("should not be called")

    q = "сравни показатели за 2021 и 2022 годы"
    out = fn(Chain(), Chain(), q, threshold=0.79)
    assert out["intent"] == "compare"
    assert out["query"] == q


def test_get_intent_hybrid_uses_llm_output_when_fast_not_confident():
    fn = FUNCS["get_intent_hybrid"]

    class JsonChain:
        def invoke(self, _):
            return {"intent": "summary", "confidence": 0.99, "query": "сделай сводку"}

    class StrChain:
        def invoke(self, _):
            return "{}"

    out = fn(JsonChain(), StrChain(), "привет", threshold=0.8)
    assert out["intent"] == "summary"
    assert out["query"] == "сделай сводку"


def test_rerank_docs_uses_ranked_ids_and_fills_tail():
    fn = FUNCS["rerank_docs"]

    class JsonChain:
        def invoke(self, _):
            return {"ranked_ids": [2, 0]}

    class StrChain:
        def invoke(self, _):
            return "{}"

    docs = [
        DummyDoc("d0", {"page": 0, "section_title": "a"}),
        DummyDoc("d1", {"page": 1, "section_title": "a"}),
        DummyDoc("d2", {"page": 2, "section_title": "a"}),
    ]
    out = fn(JsonChain(), StrChain(), "вопрос", docs, keep=3, pool=3)
    assert [d.page_content for d in out] == ["d2", "d0", "d1"]


def test_rerank_docs_fallback_to_original_order_when_invalid_ranked_ids():
    fn = FUNCS["rerank_docs"]

    class JsonChain:
        def invoke(self, _):
            return {"ranked_ids": [99, -1, "oops"]}

    class StrChain:
        def invoke(self, _):
            return "{}"

    docs = [
        DummyDoc("d0", {"page": 0, "section_title": "a"}),
        DummyDoc("d1", {"page": 1, "section_title": "a"}),
    ]
    out = fn(JsonChain(), StrChain(), "вопрос", docs, keep=2, pool=2)
    assert [d.page_content for d in out] == ["d0", "d1"]
