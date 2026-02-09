import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "eval" / "compare_runs.py"


def _write(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _run(base: Path, cand: Path, *extra: str):
    cmd = [sys.executable, str(SCRIPT), "--base", str(base), "--cand", str(cand), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)


def _sample(score: float, q1: float, q2: float, pdf="p", qs="q"):
    return {
        "summary": {"weighted_score": score, "pdf_sha256": pdf, "questions_sha256": qs},
        "results": [
            {"id": "q1", "metrics": {"score": q1}},
            {"id": "q2", "metrics": {"score": q2}},
        ],
    }


def test_compare_runs_pass(tmp_path: Path):
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    _write(b, _sample(0.50, 0.4, 0.6))
    _write(c, _sample(0.55, 0.5, 0.6))

    r = _run(b, c, "--min-delta", "0.01", "--max-regressions", "0")
    assert r.returncode == 0
    first = json.loads(r.stdout.splitlines()[0])
    assert first["passed"] is True


def test_compare_runs_fail_hash_mismatch(tmp_path: Path):
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    _write(b, _sample(0.50, 0.4, 0.6, pdf="pdfA", qs="qA"))
    _write(c, _sample(0.52, 0.5, 0.6, pdf="pdfB", qs="qA"))

    r = _run(b, c)
    assert r.returncode == 2
    first = json.loads(r.stdout.splitlines()[0])
    assert first["error"] == "hash_mismatch"


def test_compare_runs_fail_gate(tmp_path: Path):
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    _write(b, _sample(0.50, 0.5, 0.5))
    _write(c, _sample(0.49, 0.4, 0.5))

    r = _run(b, c, "--min-delta", "0.0", "--max-regressions", "0")
    assert r.returncode == 1
    first = json.loads(r.stdout.splitlines()[0])
    assert first["passed"] is False
