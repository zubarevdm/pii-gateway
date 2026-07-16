"""Интеграционные тесты HTTP-слоя через FastAPI TestClient.

Проверяем, что прокси:
  - обезличивает исходящий в upstream запрос (ПД физически не уходят);
  - восстанавливает ПД в ответе модели для пользователя.
Upstream подменяем фейком — реальный ключ не нужен.
"""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_anonymize_endpoint_roundtrip():
    text = "Клиент Петров, тел +7 900 123-45-67, почта p@mail.ru"
    r = client.post("/anonymize", json={"text": text})
    assert r.status_code == 200
    data = r.json()
    assert "+7 900 123-45-67" not in data["anonymized"]
    assert "[PHONE_1]" in data["anonymized"]
    assert data["restored"] == text
    assert data["by_type"].get("PHONE") == 1


def test_proxy_masks_outgoing_and_restores(monkeypatch):
    captured = {}

    async def fake_post(self, url, json=None, headers=None):
        # Перехватываем то, что РЕАЛЬНО уходит в upstream.
        captured["sent"] = json
        # Модель «отвечает», ссылаясь на плейсхолдер — проверим восстановление.
        user_text = json["messages"][-1]["content"]
        reply = f"Я обработал запрос: {user_text}"
        body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": json.get("model", "gpt-4o"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }],
        }
        return httpx.Response(200, json=body)

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "user",
                 "content": "Свяжись с Петровым, тел +7 900 123-45-67, p@mail.ru"}
            ],
        },
        headers={"authorization": "Bearer sk-client-key"},
    )
    assert r.status_code == 200

    # 1) В upstream ушёл ОБЕЗЛИЧЕННЫЙ текст — никаких настоящих ПД.
    sent_text = captured["sent"]["messages"][-1]["content"]
    assert "+7 900 123-45-67" not in sent_text
    assert "p@mail.ru" not in sent_text
    assert "[PHONE_1]" in sent_text and "[EMAIL_1]" in sent_text

    # 2) Пользователю вернулся ответ с ВОССТАНОВЛЕННЫМИ ПД.
    out = r.json()["choices"][0]["message"]["content"]
    assert "+7 900 123-45-67" in out
    assert "p@mail.ru" in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
