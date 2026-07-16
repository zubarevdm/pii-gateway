"""OpenAI-совместимый прокси /v1/chat/completions с обезличиванием на лету.

Поток одного запроса:
  1. Достаём текстовые части сообщений (роли user/system/tool — то, что пишет
     сотрудник; ответы assistant в истории тоже обезличиваем для связности).
  2. Один Vault на запрос: одинаковые ПД получают одинаковые плейсхолдеры во
     всех сообщениях → модель видит связный, но обезличенный диалог.
  3. Форвардим обезличенный payload в upstream (реальная модель).
  4. В ответе модели восстанавливаем плейсхолдеры обратно по Vault.
  5. Пишем аудит (факт+тип маскировки, без ПД).

Vault существует только в памяти на время запроса и уничтожается после ответа.
"""

from __future__ import annotations

import uuid

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.anonymizer import Vault
from app.detectors.base import Entity
from app.service import GatewayService, get_service

router = APIRouter()


def _anonymize_messages(
    service: GatewayService, messages: list[dict], vault: Vault
) -> list[Entity]:
    """Обезличить текстовые поля сообщений на месте. Вернуть все найденные ПД."""
    all_entities: list[Entity] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            res = service.anonymizer.anonymize(content, vault)
            msg["content"] = res.text
            all_entities.extend(res.entities)
        elif isinstance(content, list):
            # Мультимодальный формат: [{"type":"text","text":...}, ...]
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    res = service.anonymizer.anonymize(part.get("text", ""), vault)
                    part["text"] = res.text
                    all_entities.extend(res.entities)
    return all_entities


def _restore_choices(service: GatewayService, payload: dict, vault: Vault) -> None:
    """Восстановить ПД в ответе модели на месте."""
    for choice in payload.get("choices", []):
        msg = choice.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            msg["content"] = service.anonymizer.restore(msg["content"], vault)


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
    service = get_service()
    settings = service.settings
    body = await request.json()
    request_id = f"req-{uuid.uuid4().hex[:12]}"

    messages = body.get("messages", [])
    vault = Vault()
    entities = _anonymize_messages(service, messages, vault)

    # Аудит: только факт и типы, без значений.
    service.audit.record(
        request_id=request_id,
        model=str(body.get("model", "")),
        text_chars=sum(len(str(m.get("content", ""))) for m in messages),
        entities=entities,
    )

    # Заголовок авторизации: ключ клиента имеет приоритет, иначе ключ из конфига.
    auth = request.headers.get("authorization")
    if not auth and settings.upstream_api_key:
        auth = f"Bearer {settings.upstream_api_key}"
    headers = {"content-type": "application/json"}
    if auth:
        headers["authorization"] = auth

    # Стриминг буферизуем на стороне шлюза: плейсхолдер может разорваться между
    # чанками, а восстанавливать его надо целиком. Корректность важнее «настоящего»
    # стрима в MVP — отдаём результат одним SSE-событием.
    wants_stream = bool(body.get("stream"))
    upstream_body = dict(body)
    upstream_body["stream"] = False

    url = settings.upstream_base_url.rstrip("/") + "/chat/completions"
    async with httpx.AsyncClient(timeout=settings.upstream_timeout) as client:
        resp = await client.post(url, json=upstream_body, headers=headers)

    if resp.status_code != 200:
        return JSONResponse(status_code=resp.status_code, content=_safe_json(resp))

    data = resp.json()
    _restore_choices(service, data, vault)

    if not wants_stream:
        return JSONResponse(content=data)

    return StreamingResponse(_as_sse(data, request_id), media_type="text/event-stream")


def _safe_json(resp: httpx.Response) -> dict:
    try:
        return resp.json()
    except Exception:
        return {"error": {"message": resp.text[:500], "type": "upstream_error"}}


async def _as_sse(data: dict, request_id: str):
    """Завернуть готовый ответ в один SSE-чанк формата OpenAI streaming."""
    import json

    choices = []
    for ch in data.get("choices", []):
        msg = ch.get("message", {})
        choices.append({
            "index": ch.get("index", 0),
            "delta": {"role": msg.get("role", "assistant"),
                      "content": msg.get("content", "")},
            "finish_reason": ch.get("finish_reason"),
        })
    chunk = {
        "id": data.get("id", request_id),
        "object": "chat.completion.chunk",
        "model": data.get("model", ""),
        "choices": choices,
    }
    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
