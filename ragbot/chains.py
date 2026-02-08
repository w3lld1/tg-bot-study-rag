import os
from typing import Any, List

from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain_gigachat.chat_models import GigaChat
except Exception:
    from langchain_gigachat import GigaChat  # type: ignore

from ragbot.core_logic import INTENTS, detect_intent_fast, invoke_json_robust
from ragbot.text_utils import clamp_text, safe_page_range


def build_llm(settings: Any):
    key = os.environ.get("GIGA_API_KEY") or os.environ.get("GIGACHAT_API_KEY")
    if not key:
        raise RuntimeError("Не найден env GIGA_API_KEY / GIGACHAT_API_KEY")
    return GigaChat(
        credentials=key,
        verify_ssl_certs=False,
        scope="GIGACHAT_API_B2B",
        model=settings.model,
        temperature=settings.temperature,
        top_p=settings.top_p,
    )


def build_intent_chains(llm):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Верни ТОЛЬКО JSON (без markdown, без текста вокруг) в формате:\n"
            '{"intent":"definition|procedure|requirements|numbers_and_dates|compare|summary|citation_only|default",'
            '"confidence":0.0,'
            '"query":"..."}',
        ),
        ("human", "{question}"),
    ])
    return prompt | llm | JsonOutputParser(), prompt | llm | StrOutputParser()


def get_intent_hybrid(intent_chain_json, intent_chain_str, user_question: str, threshold: float):
    intent_h, conf_h = detect_intent_fast(user_question)
    q = (user_question or "").strip()
    if conf_h >= threshold and len(q) >= 18:
        return {"intent": intent_h, "confidence": conf_h, "query": q}

    fallback = {"intent": intent_h, "confidence": conf_h, "query": q}
    out = invoke_json_robust(intent_chain_json, intent_chain_str, {"question": q}, fallback, retries=1)

    intent = out.get("intent", intent_h)
    if intent not in INTENTS:
        intent = intent_h
    try:
        conf = float(out.get("confidence", conf_h))
    except Exception:
        conf = conf_h
    query = (out.get("query") or q).strip() or q
    return {"intent": intent, "confidence": conf, "query": query}


def build_multiquery_chains(llm):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Сгенерируй {n} коротких переформулировок запроса для поиска по документу.\n"
            'Верни ТОЛЬКО JSON: {"queries": ["...", "..."]}',
        ),
        ("human", "{question}"),
    ])
    return prompt | llm | JsonOutputParser(), prompt | llm | StrOutputParser()


def build_rerank_chains(llm):
    prompt = ChatPromptTemplate.from_messages([
        ("system", 'Ты — reranker. Верни ТОЛЬКО JSON: {"ranked_ids": [0,1,2]}'),
        ("human", "Вопрос: {question}\n\nФрагменты:\n{items}"),
    ])
    return prompt | llm | JsonOutputParser(), prompt | llm | StrOutputParser()


def rerank_docs(rerank_chain_json, rerank_chain_str, question: str, docs: List[Document], keep: int, pool: int) -> List[Document]:
    if not docs:
        return docs
    pool_docs = docs[:pool]
    items = []
    for i, d in enumerate(pool_docs):
        pages = safe_page_range(d.metadata)
        title = (d.metadata or {}).get("section_title", "Без заголовка")
        snippet = clamp_text(d.page_content, 340)
        items.append(f"- id:{i} | стр:{pages} | раздел:{title} | текст:{snippet}")

    fallback = {"ranked_ids": list(range(min(len(pool_docs), keep)))}
    out = invoke_json_robust(
        rerank_chain_json,
        rerank_chain_str,
        {"question": question, "items": "\n".join(items)},
        fallback,
        retries=1,
    )

    ranked_ids = []
    for x in out.get("ranked_ids", []):
        try:
            ranked_ids.append(int(x))
        except Exception:
            pass

    chosen: List[int] = []
    for rid in ranked_ids:
        if 0 <= rid < len(pool_docs) and rid not in chosen:
            chosen.append(rid)
        if len(chosen) >= keep:
            break
    if not chosen:
        return docs[:keep]

    reranked = [pool_docs[rid] for rid in chosen]
    for d in pool_docs:
        if d not in reranked:
            reranked.append(d)
        if len(reranked) >= keep:
            break
    return reranked[:keep]


def build_answer_chain_default(llm):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Ответь строго по контексту PDF. Если не найдено — 'В документе не найдено'.\n"
            "Формат:\n"
            "1) краткий ответ (1-3 предложения)\n"
            "2) Доказательства: 2-5 пунктов с (стр. X) и короткой цитатой.\n"
            "Не выдумывай. Никаких страниц XX/??.",
        ),
        ("human", "Вопрос: {question}\n\nКонтекст:\n{context}\n\nОтвет:"),
    ])
    return prompt | llm | StrOutputParser()


def build_answer_chain_summary(llm):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Дай сводку 5-10 пунктов СТРОГО по контексту PDF.\n"
            "Каждый пункт: (стр. X) + короткая цитата в кавычках (до 12 слов).\n"
            "Не добавляй фактов, которых нет в контексте.\n"
            "Никогда не пиши выдуманные страницы типа XX/??.\n"
            "Если по контексту нельзя — 'В документе не найдено'.",
        ),
        ("human", "Запрос: {question}\n\nКонтекст:\n{context}\n\nОтвет:"),
    ])
    return prompt | llm | StrOutputParser()


def build_answer_chain_compare(llm):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Сравнение: приводи ТОЛЬКО то, что явно есть в контексте.\n"
            "Формат:\n"
            "1) Короткий вывод (1-2 предложения)\n"
            "2) Сравнения пунктами (3-8 пунктов): <показатель>: 2021 -> 2022 (стр. X) — \"цитата\".\n"
            "Никогда не пиши выдуманные страницы типа XX/??.\n"
            "Если нет данных — 'В документе не найдено'.",
        ),
        ("human", "Вопрос: {question}\n\nКонтекст:\n{context}\n\nОтвет:"),
    ])
    return prompt | llm | StrOutputParser()


def build_answer_chain_citation_only(llm):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Верни 3-7 коротких цитат из контекста (каждая в кавычках) + (стр. X).\n"
            "Только цитаты, без пересказа.\n"
            "Если страницы нет в контексте — не выдумывай.\n"
            "Никаких страниц XX/??.\n"
            "Если нет — 'В документе не найдено'.",
        ),
        ("human", "Запрос: {question}\n\nКонтекст:\n{context}\n\nОтвет:"),
    ])
    return prompt | llm | StrOutputParser()


def build_answer_chain_numbers_strict(llm):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Числа/проценты/даты: давай ТОЛЬКО то, что ДОСЛОВНО в контексте.\n"
            "Для каждого значения: страница + короткая цитата (1 строка).\n"
            "Формат:\n"
            "- <показатель>: <значение> (стр. X) — \"цитата\"\n"
            "Если ничего надежного — 'В документе не найдено'.\n"
            "Никаких страниц XX/??.",
        ),
        ("human", "Вопрос: {question}\n\nКонтекст:\n{context}\n\nОтвет:"),
    ])
    return prompt | llm | StrOutputParser()
