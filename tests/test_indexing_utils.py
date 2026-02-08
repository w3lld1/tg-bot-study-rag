import ast
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import pytest


INDEXING_PATH = Path(__file__).resolve().parents[1] / "ragbot" / "indexing.py"


@dataclass
class DummyDocument:
    page_content: str
    metadata: dict


def load_indexing_symbols(*names):
    src = INDEXING_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)

    const_names = {"_HEADING_PATTERNS", "_HEADING_IGNORE"}
    selected = []

    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any(t in const_names for t in targets):
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            selected.append(node)

    mod = ast.Module(body=selected, type_ignores=[])
    code = compile(mod, filename=str(INDEXING_PATH), mode="exec")

    ns = {
        "re": __import__("re"),
        "json": json,
        "gzip": gzip,
        "Document": DummyDocument,
        "Any": Any,
        "List": List,
        "Optional": Optional,
    }
    exec(code, ns, ns)
    return {name: ns[name] for name in names}


INDEXING = load_indexing_symbols(
    "detect_heading_on_page",
    "_check_settings_compat",
    "_write_chunks_jsonl_gz",
    "_read_chunks_jsonl_gz",
)


def test_detect_heading_on_page_numeric_heading():
    fn = INDEXING["detect_heading_on_page"]
    text = "1.2 Основные положения\nТут идет содержимое страницы"
    assert fn(text) == "1.2 Основные положения"


def test_detect_heading_on_page_caps_heading():
    fn = INDEXING["detect_heading_on_page"]
    text = "ОБЗОР ПОКАЗАТЕЛЕЙ\nДальше текст"
    assert fn(text) == "ОБЗОР ПОКАЗАТЕЛЕЙ"


def test_detect_heading_on_page_ignores_table_labels():
    fn = INDEXING["detect_heading_on_page"]
    text = "ТАБЛИЦА 1\nОборот по месяцам"
    assert fn(text) is None


def test_check_settings_compat_ok():
    fn = INDEXING["_check_settings_compat"]

    class S:
        allow_index_settings_mismatch = False
        chunk_size = 1100
        chunk_overlap = 150
        embeddings_model = "emb"
        model = "llm"

    stored = {"settings": {"chunk_size": 1100, "chunk_overlap": 150, "embeddings_model": "emb", "model": "llm"}}
    fn(stored, S())  # no raise


def test_check_settings_compat_raises_on_mismatch():
    fn = INDEXING["_check_settings_compat"]

    class S:
        allow_index_settings_mismatch = False
        chunk_size = 1200
        chunk_overlap = 150
        embeddings_model = "emb"
        model = "llm"

    stored = {"settings": {"chunk_size": 1100, "chunk_overlap": 150, "embeddings_model": "emb", "model": "llm"}}
    with pytest.raises(RuntimeError):
        fn(stored, S())


def test_write_and_read_chunks_jsonl_gz_roundtrip(tmp_path):
    write_fn = INDEXING["_write_chunks_jsonl_gz"]
    read_fn = INDEXING["_read_chunks_jsonl_gz"]

    path = tmp_path / "chunks.jsonl.gz"
    docs = [
        DummyDocument(page_content="alpha", metadata={"page": 0, "source": "x"}),
        DummyDocument(page_content="beta", metadata={"page": 1, "source": "x"}),
    ]

    write_fn(str(path), docs)
    out = read_fn(str(path))

    assert len(out) == 2
    assert out[0].page_content == "alpha"
    assert out[1].metadata["page"] == 1
