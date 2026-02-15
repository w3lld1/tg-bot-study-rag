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

def test_repair_stage_numbers_prefers_numeric_evidence_when_base_not_found():
    agent = BestStableRAGAgent.__new__(BestStableRAGAgent)
    agent.answer_chain_citation_only = EchoChain()

    context = "[стр. 12] Финансовый эффект от применения AI в 2022 году составил 235 млрд руб."
    out = agent._repair_stage(
        "Какой финансовый эффект от применения AI в 2022 году?",
        "В документе не найдено.",
        context,
        "numbers_and_dates",
    )

    assert out.startswith("По релевантным фрагментам:")
    assert "стр. 12" in out
    assert "2022" in out


def test_repair_stage_numbers_drops_noisy_numeric_fallback_when_lexically_missed():
    agent = BestStableRAGAgent.__new__(BestStableRAGAgent)
    agent.answer_chain_citation_only = EchoChain()

    context = "[стр. 44] В 2022 году выбросы СО2 составили 59%."
    out = agent._repair_stage(
        "Какой финансовый эффект от применения AI в 2022 году?",
        "В документе не найдено.",
        context,
        "numbers_and_dates",
    )

    assert out == "В документе не найдено."


def test_numeric_terminal_guard_blocks_fragments_fallback():
    agent = BestStableRAGAgent.__new__(BestStableRAGAgent)

    out = agent._enforce_numbers_terminal_quality(
        "Найденные релевантные фрагменты:\n- (стр. 12) \"...\""
    )

    assert out["guard_applied"] is True
    assert out["answer"] == "В документе не найдено."
    assert out["reason"] == "fragments_fallback"


def test_repair_stage_targeted_mission_repair():
    agent = BestStableRAGAgent.__new__(BestStableRAGAgent)
    agent.answer_chain_citation_only = EchoChain()

    context = "[стр. 7] Миссия Мы даем людям уверенность и надежность, мы делаем их жизнь лучше."
    out = agent._repair_stage("Как сформулирована миссия Сбера?", "В документе не найдено.", context, "default")

    assert "Мы даем людям уверенность" in out
    assert "стр. 7" in out


def test_repair_stage_targeted_risk_sections_repair():
    agent = BestStableRAGAgent.__new__(BestStableRAGAgent)
    agent.answer_chain_citation_only = EchoChain()

    context = "[стр. 73-76] ОТЧЕТ ПО РИСКАМ Система управления рисками.\n\n[стр. 77-82] Подход к управлению ключевыми рисками группы."
    out = agent._repair_stage("Какие ключевые разделы включает часть отчета по рискам?", "В документе не найдено.", context, "default")

    assert "Система управления рисками" in out
    assert "Подходы к управлению" in out
