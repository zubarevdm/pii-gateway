"""Точка входа FastAPI-шлюза.

Запуск:
    uvicorn app.main:app --host 127.0.0.1 --port 8787

Эндпоинты:
    POST /v1/chat/completions  — OpenAI-совместимый прокси с обезличиванием
    POST /anonymize            — отладка/демо: показать обезличивание без upstream
    GET  /health               — проверка живости
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from app.anonymizer import Vault
from app.proxy.chat import router as chat_router
from app.service import get_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Прогреваем NER-модель на старте, чтобы первый реальный запрос не ждал.
    get_service().warmup()
    yield


app = FastAPI(title="PII-шлюз перед LLM", version="0.1.0", lifespan=lifespan)
app.include_router(chat_router)


@app.get("/health")
async def health() -> dict:
    svc = get_service()
    return {"status": "ok", "ner": svc.settings.enable_ner}


class AnonymizeIn(BaseModel):
    text: str


class AnonymizeOut(BaseModel):
    anonymized: str
    restored: str
    replaced: int
    by_type: dict[str, int]


@app.post("/anonymize", response_model=AnonymizeOut)
async def anonymize(inp: AnonymizeIn) -> AnonymizeOut:
    """Демонстрация: обезличить текст и тут же восстановить (round-trip).

    Удобно для проверки качества детекции без ключа к модели. Оригинальные ПД
    остаются только в этом ответе локально — наружу ничего не уходит.
    """
    svc = get_service()
    vault = Vault()
    res = svc.anonymizer.anonymize(inp.text, vault)
    restored = svc.anonymizer.restore(res.text, vault)
    by_type: dict[str, int] = {}
    for e in res.entities:
        by_type[e.type.value] = by_type.get(e.type.value, 0) + 1
    return AnonymizeOut(
        anonymized=res.text,
        restored=restored,
        replaced=res.replaced,
        by_type=by_type,
    )
