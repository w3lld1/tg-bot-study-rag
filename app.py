# app.py
# Single-file TG bot + RAG agent (VM-ready)
# - Polling bot (aiogram v3)
# - Per-user PDF indexing (FAISS + BM25)
# - /reset clears user session (index)
# - Secrets: env BOT_TOKEN + GIGACHAT_API_KEY (preferred),
#   fallback: Google Secret Manager (optional)

import os
import shutil
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, Optional

# -----------------------------
# Logging
# -----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("tg-rag-bot")


# -----------------------------
# Secrets (env first, GSM fallback)
# -----------------------------
def get_env(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v:
        v = v.strip()
    return v or None

def load_secret_from_gsm(secret_id: str, project_id: Optional[str] = None) -> Optional[str]:
    """
    Optional fallback: read secret from Google Secret Manager.
    Requires:
      pip install google-cloud-secret-manager
    And VM SA must have roles/secretmanager.secretAccessor.
    """
    try:
        from google.cloud import secretmanager  # type: ignore
    except Exception:
        return None

    project_id = project_id or os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        return None

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    resp = client.access_secret_version(request={"name": name})
    return resp.payload.data.decode("utf-8").strip()

def must_get_secret(name: str) -> str:
    # Prefer env
    v = get_env(name)
    if v:
        return v

    # Fallback GSM with same secret name
    v = load_secret_from_gsm(name)
    if v:
        return v

    raise RuntimeError(
        f"Missing secret {name}. Provide env var {name} (preferred) or enable GSM fallback."
    )


# -----------------------------
# Agent code (your version, only paths made session-aware)
# -----------------------------
AGENT_VERSION = "v_final_patched_qos_1_4_plus_vm"

from langchain_core.documents import Document  # noqa: E402

from ragbot.text_utils import ensure_dir  # noqa: E402
from ragbot.indexing import ingest_pdf  # noqa: E402
from ragbot.agent import BestStableRAGAgent  # noqa: E402


@dataclass(frozen=True)
class Settings:
    # Base data dir (mounted volume in docker)
    data_dir: str = "./data"

    # chunking
    chunk_size: int = 1100
    chunk_overlap: int = 150

    # retrieval (base)
    bm25_k: int = 14
    faiss_k: int = 14

    # multiquery
    multiquery_enabled: bool = True
    multiquery_n: int = 2
    multiquery_parallel: bool = True
    multiquery_threads: int = 6
    multiquery_k_per_query: int = 12

    # separate multiquery count for numbers/dates
    numbers_multiquery_n: int = 4

    # rerank
    rerank_enabled: bool = True
    rerank_pool: int = 18
    rerank_keep: int = 10

    # context
    final_docs_limit: int = 10
    max_context_chars: int = 16000
    diversify_max_per_group: int = 3

    # intent
    llm_intent_threshold: float = 0.78

    # llm
    temperature: float = 0.2
    top_p: float = 0.9
    model: str = "GigaChat-2-Max"

    # embeddings
    embeddings_model: str = "GigaEmbeddings-3B-2025-09"
    query_embedding_cache_size: int = 512

    # safety / compatibility
    allow_dangerous_faiss_deserialization: bool = False
    allow_index_settings_mismatch: bool = False

    # ---- Quality-over-speed knobs (universal) ----
    numbers_k_multiplier: int = 3
    numbers_neighbors_window: int = 2
    numbers_disable_diversify: bool = True

    # Second pass
    second_pass_enabled: bool = True
    second_pass_multiquery_n: int = 6
    second_pass_k_multiplier: int = 5




# -----------------------------
# Session storage (per user)
# -----------------------------
def user_root_dir(settings: Settings, user_id: int) -> str:
    return os.path.join(settings.data_dir, "users", str(user_id))

def user_index_dir(settings: Settings, user_id: int) -> str:
    return os.path.join(user_root_dir(settings, user_id), "rag_index")

def user_pdf_path(settings: Settings, user_id: int) -> str:
    return os.path.join(user_root_dir(settings, user_id), "document.pdf")

def user_has_index(settings: Settings, user_id: int) -> bool:
    idx = user_index_dir(settings, user_id)
    return os.path.exists(os.path.join(idx, "meta.json")) and os.path.exists(os.path.join(idx, "chunks.jsonl.gz"))

def reset_user_session(settings: Settings, user_id: int) -> None:
    root = user_root_dir(settings, user_id)
    if os.path.exists(root):
        shutil.rmtree(root, ignore_errors=True)


# -----------------------------
# Telegram bot (aiogram polling)
# -----------------------------
from aiogram import Bot, Dispatcher, F  # noqa: E402
from aiogram.filters import Command  # noqa: E402
from aiogram.types import Message, Document as TgDocument  # noqa: E402
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError


BOT_TOKEN = must_get_secret("BOT_TOKEN")
GIGACHAT_API_KEY = must_get_secret("GIGACHAT_API_KEY")

# Expose to agent builders
os.environ["GIGACHAT_API_KEY"] = GIGACHAT_API_KEY

DATA_DIR = os.getenv("DATA_DIR", "./data")
SETTINGS = Settings(data_dir=DATA_DIR)

dp = Dispatcher()


# Keep agent instances in-memory (per user) to avoid reload for every message
# If container restarts — will be rebuilt from on-disk index.
AGENTS: Dict[int, BestStableRAGAgent] = {}
LOCKS: Dict[int, asyncio.Lock] = {}


def get_user_lock(user_id: int) -> asyncio.Lock:
    if user_id not in LOCKS:
        LOCKS[user_id] = asyncio.Lock()
    return LOCKS[user_id]

def get_or_build_agent(user_id: int) -> BestStableRAGAgent:
    if user_id in AGENTS:
        return AGENTS[user_id]
    if not user_has_index(SETTINGS, user_id):
        raise RuntimeError("NO_INDEX")
    agent = BestStableRAGAgent(SETTINGS, index_dir=user_index_dir(SETTINGS, user_id), agent_version=AGENT_VERSION)
    AGENTS[user_id] = agent
    return agent


@dp.message(Command("start"))
async def cmd_start(msg: Message):
    text = (
        "Привет! Пришли мне PDF, я построю индекс и буду отвечать на вопросы по документу.\n\n"
        "Команды:\n"
        "/reset — очистить текущий документ и начать заново\n"
        "/status — статус (есть ли документ)\n\n"
        "После загрузки PDF просто задавай вопросы текстом."
    )
    await msg.answer(text)

@dp.message(Command("status"))
async def cmd_status(msg: Message):
    uid = msg.from_user.id
    has = user_has_index(SETTINGS, uid)
    await msg.answer("✅ Документ загружен и проиндексирован." if has else "❌ Документ не загружен. Пришли PDF.")

@dp.message(Command("reset"))
async def cmd_reset(msg: Message):
    uid = msg.from_user.id
    async with get_user_lock(uid):
        reset_user_session(SETTINGS, uid)
        AGENTS.pop(uid, None)
    await msg.answer("Ок. Сессия очищена. Пришли новый PDF.")

def is_pdf(doc: TgDocument) -> bool:
    if doc.mime_type == "application/pdf":
        return True
    if (doc.file_name or "").lower().endswith(".pdf"):
        return True
    return False

@dp.message(F.document)
async def on_pdf(msg: Message, bot: Bot):
    uid = msg.from_user.id
    doc: TgDocument = msg.document

    if not doc or not is_pdf(doc):
        await msg.answer("Пришли, пожалуйста, PDF-файл.")
        return

    # Можно заранее проверить размер, если Telegram прислал file_size
    # (не всегда помогает, но иногда сразу даёт понятное сообщение)
    max_hint_mb = 20
    if getattr(doc, "file_size", None) and doc.file_size > max_hint_mb * 1024 * 1024:
        await msg.answer(
            f"⚠️ Файл выглядит большим (~{doc.file_size/1024/1024:.1f} MB). "
            "Telegram Bot API может не дать его скачать. Если не получится — сожми/разбей PDF."
        )

    async with get_user_lock(uid):
        # New doc → wipe old session
        reset_user_session(SETTINGS, uid)
        AGENTS.pop(uid, None)

        ensure_dir(user_root_dir(SETTINGS, uid))
        pdf_path = user_pdf_path(SETTINGS, uid)

        await msg.answer("📥 Скачиваю PDF...")

        # Скачивание
        try:
            # Получаем file_path (может упасть на больших файлах: "file is too big")
            file = await bot.get_file(doc.file_id)

            # Таймаут на скачивание, чтобы не зависать бесконечно
            await asyncio.wait_for(
                bot.download_file(file.file_path, destination=pdf_path),
                timeout=180,
            )

        except TelegramBadRequest as e:
            # Самый частый кейс у тебя: Telegram не отдаёт file_path
            if "file is too big" in str(e).lower():
                await msg.answer(
                    "❌ PDF слишком большой для загрузки через Telegram Bot API.\n\n"
                    "Что можно сделать:\n"
                    "1) Сжать PDF (уменьшить качество/картинки)\n"
                    "2) Разбить на несколько частей и прислать по очереди\n"
                    "3) Прислать текстовую версию (без сканов), если есть\n\n"
                    "После этого пришли PDF заново."
                )
                return

            await msg.answer(
                f"❌ Telegram вернул ошибку при загрузке файла: {e}\n"
                "Попробуй прислать PDF ещё раз или /reset."
            )
            return

        except asyncio.TimeoutError:
            await msg.answer(
                "❌ Не удалось скачать PDF: таймаут.\n"
                "Попробуй ещё раз, либо сожми файл/разбей на части."
            )
            return

        except TelegramNetworkError as e:
            await msg.answer(
                "❌ Сетевая ошибка при скачивании PDF.\n"
                "Попробуй ещё раз через минуту."
            )
            return

        except Exception as e:
            logger.exception("PDF download failed: %s", e)
            await msg.answer("❌ Не удалось скачать PDF из Telegram. Попробуй ещё раз или /reset.")
            return

        # Базовая проверка, что файл реально появился и не пустой
        try:
            if (not os.path.exists(pdf_path)) or os.path.getsize(pdf_path) < 1024:
                await msg.answer(
                    "❌ Файл скачался некорректно (пустой/повреждён). "
                    "Попробуй прислать PDF ещё раз или сжать его."
                )
                return
        except Exception:
            pass

        await msg.answer("🔎 Индексирую… Это может занять 1–5 минут (зависит от PDF).")

        # Run ingest in thread to avoid blocking event loop
        idx_dir = user_index_dir(SETTINGS, uid)

        def _do_ingest():
            ingest_pdf(pdf_path, SETTINGS, index_dir=idx_dir, agent_version=AGENT_VERSION)

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, _do_ingest)
        except Exception as e:
            logger.exception("ingest failed: %s", e)
            await msg.answer(
                "❌ Ошибка при индексировании PDF.\n"
                "Попробуй другой PDF или /reset."
            )
            return

        # Build agent and cache it
        try:
            AGENTS[uid] = BestStableRAGAgent(SETTINGS, index_dir=idx_dir, agent_version=AGENT_VERSION)
        except Exception as e:
            logger.exception("agent build failed: %s", e)
            await msg.answer(
                "❌ Индекс построен, но не удалось поднять агента.\n"
                "Попробуй /reset и загрузить PDF заново."
            )
            return

    await msg.answer("✅ Готово! Теперь задавай вопросы по документу.")

@dp.message(F.text)
async def on_text(msg: Message):
    uid = msg.from_user.id
    q = (msg.text or "").strip()
    if not q:
        return

    async with get_user_lock(uid):
        try:
            agent = get_or_build_agent(uid)
        except RuntimeError as e:
            if str(e) == "NO_INDEX":
                await msg.answer("Сначала пришли PDF-файл, чтобы я построил индекс.")
                return
            raise

        await msg.answer("⏳ Думаю...")

        # Run LLM in executor (langchain sync calls)
        loop = asyncio.get_running_loop()
        def _ask():
            return agent.ask(q)

        try:
            ans = await loop.run_in_executor(None, _ask)
        except Exception as e:
            logger.exception("ask failed: %s", e)
            await msg.answer("Ошибка при обработке запроса. Попробуй ещё раз или /reset.")
            return

    # Telegram message limit ~4096
    if len(ans) <= 3800:
        await msg.answer(ans)
    else:
        # split
        for part in [ans[i:i+3800] for i in range(0, len(ans), 3800)]:
            await msg.answer(part)


async def main():
    bot = Bot(token=BOT_TOKEN)
    logger.info("Bot starting (polling). data_dir=%s", SETTINGS.data_dir)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
