# app.py
# Single-file TG bot + RAG agent (VM-ready)
# - Polling bot (aiogram v3)
# - Per-user PDF indexing (FAISS + BM25)
# - /reset clears user session (index)
# - Secrets: env BOT_TOKEN + GIGACHAT_API_KEY (preferred),
#   fallback: Google Secret Manager (optional)

import os
import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

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

from ragbot.bot import run_polling  # noqa: E402


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




BOT_TOKEN = must_get_secret("BOT_TOKEN")
GIGACHAT_API_KEY = must_get_secret("GIGACHAT_API_KEY")

# Expose to agent/index builders
os.environ["GIGACHAT_API_KEY"] = GIGACHAT_API_KEY

DATA_DIR = os.getenv("DATA_DIR", "./data")
SETTINGS = Settings(data_dir=DATA_DIR)


async def main():
    await run_polling(
        bot_token=BOT_TOKEN,
        settings=SETTINGS,
        agent_version=AGENT_VERSION,
        logger=logger,
    )


if __name__ == "__main__":
    asyncio.run(main())
