from ragbot.agent import BestStableRAGAgent


class EchoChain:
    def __init__(self):
        self.last_payload = None

    def invoke(self, payload):
        self.last_payload = payload
        return "ok"


def test_answer_stage_uses_synthesis_context_for_default_intent():
    agent = BestStableRAGAgent.__new__(BestStableRAGAgent)
    agent.answer_chain_default = EchoChain()
    agent.answer_chain_summary = EchoChain()
    agent.answer_chain_compare = EchoChain()
    agent.answer_chain_citation_only = EchoChain()
    agent.answer_chain_numbers = EchoChain()

    plan = {"synthesis_context": "PLANNED-CONTEXT"}
    out = agent._answer_stage("default", "вопрос", "RAW-CONTEXT", plan=plan)

    assert out == "ok"
    assert agent.answer_chain_default.last_payload["context"] == "PLANNED-CONTEXT"


def test_answer_stage_keeps_raw_context_for_citation_only():
    agent = BestStableRAGAgent.__new__(BestStableRAGAgent)
    agent.answer_chain_default = EchoChain()
    agent.answer_chain_summary = EchoChain()
    agent.answer_chain_compare = EchoChain()
    agent.answer_chain_citation_only = EchoChain()
    agent.answer_chain_numbers = EchoChain()

    plan = {"synthesis_context": "PLANNED-CONTEXT"}
    out = agent._answer_stage("citation_only", "вопрос", "RAW-CONTEXT", plan=plan)

    assert out == "ok"
    assert agent.answer_chain_citation_only.last_payload["context"] == "RAW-CONTEXT"

def test_repair_stage_numbers_returns_not_found_when_base_not_found():
    agent = BestStableRAGAgent.__new__(BestStableRAGAgent)
    agent.answer_chain_citation_only = EchoChain()

    context = "[стр. 12] Финансовый эффект от применения AI в 2022 году ..."
    out = agent._repair_stage("вопрос", "В документе не найдено.", context, "numbers_and_dates")

    assert out == "В документе не найдено."


def test_numeric_terminal_guard_blocks_fragments_fallback():
    agent = BestStableRAGAgent.__new__(BestStableRAGAgent)

    out = agent._enforce_numbers_terminal_quality(
        "Найденные релевантные фрагменты:\n- (стр. 12) \"...\""
    )

    assert out["guard_applied"] is True
    assert out["answer"] == "В документе не найдено."
    assert out["reason"] == "fragments_fallback"
