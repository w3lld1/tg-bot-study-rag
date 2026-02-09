import ast
import re
from pathlib import Path


RUN_EVAL_PATH = Path(__file__).resolve().parents[1] / "eval" / "run_eval.py"


def load_symbols(*names):
    src = RUN_EVAL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    selected = [n for n in tree.body if isinstance(n, ast.Assign) and any(getattr(t, 'id', None) == 'CITATION_PAT' for t in n.targets)]
    selected += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]

    mod = ast.Module(body=selected, type_ignores=[])
    code = compile(mod, filename=str(RUN_EVAL_PATH), mode="exec")
    ns = {"re": re, "Any": object}
    exec(code, ns, ns)
    return {name: ns[name] for name in names}


F = load_symbols("_normalize_text", "_groups_from_question", "score_answer")


def test_normalize_text_numbers_and_comparators():
    fn = F["_normalize_text"]
    s = "Более 1 234,5 и Ёж"
    out = fn(s)
    assert ">1234.5" in out
    assert "еж" in out


def test_groups_from_question_includes_must_include_any():
    fn = F["_groups_from_question"]
    q = {"must_include": ["a"], "must_include_any": [["b1", "b2"], "c"]}
    out = fn(q)
    assert out == [["a"], ["b1", "b2"], ["c"]]


def test_score_answer_with_or_groups_and_citation_penalty():
    fn = F["score_answer"]
    q = {
        "must_include": ["270,5"],
        "must_include_any": [["78,3%", "78.3%"]],
        "must_not_include": ["xx"],
        "require_citation": True,
    }

    # include hit есть, но ссылки нет => штраф
    out1 = fn("Чистая прибыль 270.5 и снижение 78.3%", q)
    assert out1["include_hits"] == 2
    assert out1["citation_ok"] is False
    assert out1["citation_penalty"] == 0.2

    # include hit + ссылка => без штрафа
    out2 = fn("Чистая прибыль 270,5 и снижение 78,3% (стр. 62)", q)
    assert out2["citation_ok"] is True
    assert out2["score"] > out1["score"]
