import ast
from pathlib import Path


BOT_PATH = Path(__file__).resolve().parents[1] / "ragbot" / "bot.py"


def load_bot_symbols(*names):
    src = BOT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)

    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]

    mod = ast.Module(body=selected, type_ignores=[])
    code = compile(mod, filename=str(BOT_PATH), mode="exec")
    ns = {"os": __import__("os"), "Any": object, "TgDocument": object}
    exec(code, ns, ns)
    return {name: ns[name] for name in names}


FUNCS = load_bot_symbols("user_root_dir", "user_index_dir", "user_pdf_path", "user_has_index", "is_pdf")


class S:
    data_dir = "/tmp/data"


class Doc:
    def __init__(self, mime_type=None, file_name=None):
        self.mime_type = mime_type
        self.file_name = file_name


def test_user_paths():
    assert FUNCS["user_root_dir"](S(), 42) == "/tmp/data/users/42"
    assert FUNCS["user_index_dir"](S(), 42).endswith("/tmp/data/users/42/rag_index")
    assert FUNCS["user_pdf_path"](S(), 42).endswith("/tmp/data/users/42/document.pdf")


def test_user_has_index(tmp_path):
    class S2:
        data_dir = str(tmp_path)

    idx = tmp_path / "users" / "7" / "rag_index"
    idx.mkdir(parents=True)
    (idx / "meta.json").write_text("{}", encoding="utf-8")
    (idx / "chunks.jsonl.gz").write_text("x", encoding="utf-8")

    assert FUNCS["user_has_index"](S2(), 7) is True


def test_is_pdf_by_mime_or_extension():
    assert FUNCS["is_pdf"](Doc(mime_type="application/pdf", file_name="x.bin")) is True
    assert FUNCS["is_pdf"](Doc(mime_type="application/octet-stream", file_name="report.PDF")) is True
    assert FUNCS["is_pdf"](Doc(mime_type="image/png", file_name="img.png")) is False
