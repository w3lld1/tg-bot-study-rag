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
