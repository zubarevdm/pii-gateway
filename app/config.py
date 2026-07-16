"""Конфигурация шлюза. Читается из переменных окружения / .env.

ВАЖНО: ключ к зарубежной модели (OPENAI_API_KEY и т.п.) — это доступ САМОГО
клиента к его LLM. Шлюз лишь форвардит запрос. Мы не платим за токены клиента
и не храним его ключ дольше времени запроса.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8",
                                      extra="ignore")

    # --- куда форвардим обезличенный запрос ---
    # OpenAI-совместимый upstream. По умолчанию — OpenAI; можно указать любой
    # совместимый endpoint (в т.ч. свой прокси-доступ к Anthropic/др.).
    upstream_base_url: str = "https://api.openai.com/v1"
    # Ключ можно задать тут (тогда клиент шлёт запрос без своего ключа), либо
    # оставить пустым и пробрасывать Authorization клиента «как есть».
    upstream_api_key: str = ""
    upstream_timeout: float = 120.0

    # --- сервер шлюза ---
    host: str = "127.0.0.1"
    port: int = 8787

    # --- детекция ---
    enable_ner: bool = True          # слой Natasha (можно отключить для скорости)
    ner_confidence: float = 0.85

    # --- аудит ---
    audit_enabled: bool = True
    audit_path: str = "./audit.log"  # пишет ТОЛЬКО факт+тип маскировки, без ПД


def get_settings() -> Settings:
    return Settings()
