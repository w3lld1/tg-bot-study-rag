import json
import subprocess
import sys
from pathlib import Path

from eval.benchmark.gate_utils import GateConfig, evaluate_multi_run_gate

REPO = Path(__file__).resolve().parents[1]
RUN_BENCH = REPO / "eval" / "benchmark" / "run_benchmark.py"
GATE_SUMMARY = REPO / "eval" / "benchmark" / "gate_summary.py"


def test_gate_utils_checks_std_and_regressions():
    cand_summary = {
        "weighted_score": 0.5,
        "weighted_score_stats": {"mean": 0.5, "std": 0.02},
        "question_weighted_score": 0.49,
        "question_weighted_score_stats": {"mean": 0.49, "std": 0.03},
    }
    manifest = [
        {"status": "ok", "weighted_score": 0.50},
        {"status": "ok", "weighted_score": 0.48},
        {"status": "ok", "weighted_score": 0.52},
    ]
    baseline = {"weighted_score": 0.51}

    res = evaluate_multi_run_gate(
        candidate_summary=cand_summary,
        run_manifest=manifest,
        baseline_summary=baseline,
        gate=GateConfig(
            min_delta=0.0,
            max_regressions=0,
            max_std_weighted=0.01,
            max_std_question=0.02,
            max_drop_vs_baseline=0.005,
        ),
    )

    assert res["passed"] is False
    assert len(res["reasons"]) >= 3


def _write_bad_config(path: Path):
    path.write_text(
        json.dumps(
            {
                "name": "test-benchmark",
                "datasets": [
                    {
                        "id": "bad",
                        "pdf": str(path.parent / "missing.pdf"),
                        "questions": str(path.parent / "missing.json"),
                        "weight": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_run_manifest_fail_fast_vs_best_effort(tmp_path: Path):
    cfg = tmp_path / "cfg.json"
    _write_bad_config(cfg)

    out_ff = tmp_path / "out_ff.json"
    cmd_ff = [
        sys.executable,
        str(RUN_BENCH),
        "--config",
        str(cfg),
        "--out",
        str(out_ff),
        "--runs",
        "3",
        "--run-mode",
        "fail-fast",
    ]
    r_ff = subprocess.run(cmd_ff, cwd=REPO, capture_output=True, text=True)
    assert r_ff.returncode == 2
    ff = json.loads(out_ff.read_text(encoding="utf-8"))
    assert len(ff["run_manifest"]) == 1
    assert ff["run_manifest"][0]["status"] == "infra_error"
    assert ff["run_manifest"][0]["error"]
    assert ff["run_manifest"][0]["elapsed_sec"] is not None

    out_be = tmp_path / "out_be.json"
    cmd_be = [
        sys.executable,
        str(RUN_BENCH),
        "--config",
        str(cfg),
        "--out",
        str(out_be),
        "--runs",
        "3",
        "--run-mode",
        "best-effort",
    ]
    r_be = subprocess.run(cmd_be, cwd=REPO, capture_output=True, text=True)
    assert r_be.returncode == 2
    be = json.loads(out_be.read_text(encoding="utf-8"))
    assert len(be["run_manifest"]) == 3
    assert all(x["status"] == "infra_error" for x in be["run_manifest"])


def test_gate_summary_cli(tmp_path: Path):
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    base.write_text(json.dumps({"summary": {"weighted_score": 0.7}}), encoding="utf-8")
    cand.write_text(
        json.dumps(
            {
                "summary": {
                    "weighted_score": 0.68,
                    "weighted_score_stats": {"mean": 0.68, "std": 0.02},
                    "question_weighted_score_stats": {"mean": 0.68, "std": 0.02},
                },
                "run_manifest": [
                    {"status": "ok", "weighted_score": 0.68},
                    {"status": "ok", "weighted_score": 0.67},
                ],
            }
        ),
        encoding="utf-8",
    )

    cmd = [
        sys.executable,
        str(GATE_SUMMARY),
        "--summary",
        str(cand),
        "--baseline",
        str(base),
        "--max-drop-vs-baseline",
        "0.01",
        "--max-regressions",
        "0",
    ]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 1
