import asyncio
import logging
import os
import shutil
from typing import Any, Dict

from aiogram import Bot, Dispatcher, F
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.filters import Command
from aiogram.types import Document as TgDocument
from aiogram.types import Message

from ragbot.agent import BestStableRAGAgent
from ragbot.indexing import ingest_pdf
from ragbot.text_utils import ensure_dir


def user_root_dir(settings: Any, user_id: int) -> str:
    return os.path.join(settings.data_dir, "users", str(user_id))


def user_index_dir(settings: Any, user_id: int) -> str:
    return os.path.join(user_root_dir(settings, user_id), "rag_index")


def user_pdf_path(settings: Any, user_id: int) -> str:
    return os.path.join(user_root_dir(settings, user_id), "document.pdf")


def user_has_index(settings: Any, user_id: int) -> bool:
    idx = user_index_dir(settings, user_id)
    return os.path.exists(os.path.join(idx, "meta.json")) and os.path.exists(os.path.join(idx, "chunks.jsonl.gz"))


def reset_user_session(settings: Any, user_id: int) -> None:
    root = user_root_dir(settings, user_id)
    if os.path.exists(root):
        shutil.rmtree(root, ignore_errors=True)


def is_pdf(doc: TgDocument) -> bool:
    if doc.mime_type == "application/pdf":
        return True
    if (doc.file_name or "").lower().endswith(".pdf"):
        return True
    return False


def build_dispatcher(settings: Any, agent_version: str, logger: logging.Logger) -> Dispatcher:
    dp = Dispatcher()

    agents: Dict[int, BestStableRAGAgent] = {}
    locks: Dict[int, asyncio.Lock] = {}

    def get_user_lock(user_id: int) -> asyncio.Lock:
        if user_id not in locks:
            locks[user_id] = asyncio.Lock()
        return locks[user_id]

    def get_or_build_agent(user_id: int) -> BestStableRAGAgent:
        if user_id in agents:
            return agents[user_id]
        if not user_has_index(settings, user_id):
            raise RuntimeError("NO_INDEX")
        agent = BestStableRAGAgent(settings, index_dir=user_index_dir(settings, user_id), agent_version=agent_version)
        agents[user_id] = agent
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
        has = user_has_index(settings, uid)
        await msg.answer("✅ Документ загружен и проиндексирован." if has else "❌ Документ не загружен. Пришли PDF.")

    @dp.message(Command("reset"))
    async def cmd_reset(msg: Message):
        uid = msg.from_user.id
        async with get_user_lock(uid):
            reset_user_session(settings, uid)
            agents.pop(uid, None)
        await msg.answer("Ок. Сессия очищена. Пришли новый PDF.")

    @dp.message(F.document)
    async def on_pdf(msg: Message, bot: Bot):
        uid = msg.from_user.id
        doc: TgDocument = msg.document

        if not doc or not is_pdf(doc):
            await msg.answer("Пришли, пожалуйста, PDF-файл.")
            return

        max_hint_mb = 20
        if getattr(doc, "file_size", None) and doc.file_size > max_hint_mb * 1024 * 1024:
            await msg.answer(
                f"⚠️ Файл выглядит большим (~{doc.file_size/1024/1024:.1f} MB). "
                "Telegram Bot API может не дать его скачать. Если не получится — сожми/разбей PDF."
            )

        async with get_user_lock(uid):
            reset_user_session(settings, uid)
            agents.pop(uid, None)

            ensure_dir(user_root_dir(settings, uid))
            pdf_path = user_pdf_path(settings, uid)

            await msg.answer("📥 Скачиваю PDF...")

            try:
                file = await bot.get_file(doc.file_id)
                await asyncio.wait_for(
                    bot.download_file(file.file_path, destination=pdf_path),
                    timeout=180,
                )

            except TelegramBadRequest as e:
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

            except TelegramNetworkError:
                await msg.answer(
                    "❌ Сетевая ошибка при скачивании PDF.\n"
                    "Попробуй ещё раз через минуту."
                )
                return

            except Exception as e:
                logger.exception("PDF download failed: %s", e)
                await msg.answer("❌ Не удалось скачать PDF из Telegram. Попробуй ещё раз или /reset.")
                return

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

            idx_dir = user_index_dir(settings, uid)

            def _do_ingest():
                ingest_pdf(pdf_path, settings, index_dir=idx_dir, agent_version=agent_version)

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

            try:
                agents[uid] = BestStableRAGAgent(settings, index_dir=idx_dir, agent_version=agent_version)
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

            loop = asyncio.get_running_loop()

            def _ask():
                return agent.ask(q)

            try:
                ans = await loop.run_in_executor(None, _ask)
            except Exception as e:
                logger.exception("ask failed: %s", e)
                await msg.answer("Ошибка при обработке запроса. Попробуй ещё раз или /reset.")
                return

        if len(ans) <= 3800:
            await msg.answer(ans)
        else:
            for part in [ans[i : i + 3800] for i in range(0, len(ans), 3800)]:
                await msg.answer(part)

    return dp


async def run_polling(bot_token: str, settings: Any, agent_version: str, logger: logging.Logger) -> None:
    bot = Bot(token=bot_token)
    dp = build_dispatcher(settings=settings, agent_version=agent_version, logger=logger)
    logger.info("Bot starting (polling). data_dir=%s", settings.data_dir)
    await dp.start_polling(bot)
