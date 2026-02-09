import ast
import re
from pathlib import Path


AGENT_PATH = Path(__file__).resolve().parents[1] / "ragbot" / "agent.py"


def load_symbols(*names):
    src = AGENT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)

    selected = []
    for n in tree.body:
        if isinstance(n, ast.Assign):
            if any(getattr(t, "id", None) == "_CITATION_RE" for t in n.targets):
                selected.append(n)
        if isinstance(n, ast.FunctionDef) and n.name in names:
            selected.append(n)

    mod = ast.Module(body=selected, type_ignores=[])
    code = compile(mod, filename=str(AGENT_PATH), mode="exec")
    ns = {"re": re}
    exec(code, ns, ns)
    return {name: ns[name] for name in names}


F = load_symbols("_has_citation")


def test_has_citation_detects_parenthesized_page_ref():
    fn = F["_has_citation"]
    assert fn('Ответ ... (стр. 12) "цитата"') is True


def test_has_citation_detects_plain_page_ref():
    fn = F["_has_citation"]
    assert fn("Ответ... стр. 7") is True


def test_has_citation_false_when_no_page_ref():
    fn = F["_has_citation"]
    assert fn("Ответ без ссылок") is False
