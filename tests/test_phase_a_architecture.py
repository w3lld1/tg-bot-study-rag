from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = (ROOT / "ragbot" / "agent.py").read_text(encoding="utf-8")
EVAL_SRC = (ROOT / "eval" / "run_eval.py").read_text(encoding="utf-8")


def test_agent_has_staged_pipeline_methods():
    assert "def _intent_stage(" in AGENT_SRC
    assert "def _retrieve_with_trace(" in AGENT_SRC
    assert "def _answer_stage(" in AGENT_SRC
    assert "def _validation_stage(" in AGENT_SRC
    assert "def _repair_stage(" in AGENT_SRC


def test_agent_exposes_last_trace():
    assert "self._last_trace" in AGENT_SRC
    assert "def get_last_trace(" in AGENT_SRC


def test_eval_report_contains_trace_field():
    assert '"trace": agent.get_last_trace()' in EVAL_SRC
