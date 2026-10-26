"""Список моделей, доступных по ключу VseLLM (или другому OpenAI-совместимому шлюзу).

    python scripts/list_models.py              # все модели
    python scripts/list_models.py gpt claude   # только те, в имени которых есть подстрока

Читает VSELLM_API_KEY и VSELLM_BASE_URL из .env; выбранное имя вставьте в VSELLM_MODEL_NAME.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

from agent_module.llms import DEFAULT_VSELLM_BASE_URL, _is_placeholder  # noqa: E402


def main() -> None:
    load_dotenv(ROOT / ".env", override=False)
    key = os.getenv("VSELLM_API_KEY", "")
    if _is_placeholder(key):
        raise SystemExit("Задайте VSELLM_API_KEY в .env")
    base_url = (os.getenv("VSELLM_BASE_URL") or "").strip() or DEFAULT_VSELLM_BASE_URL
    from openai import OpenAI

    try:
        models = sorted(m.id for m in OpenAI(api_key=key, base_url=base_url).models.list().data)
    except Exception as exc:
        raise SystemExit(f"Не удалось получить список моделей с {base_url}: {type(exc).__name__}: {exc}")
    wanted = [w.lower() for w in sys.argv[1:]]
    shown = [m for m in models if not wanted or any(w in m.lower() for w in wanted)]
    print(f"{base_url}: моделей {len(models)}, показано {len(shown)}")
    for model in shown:
        print(f"  {model}")


if __name__ == "__main__":
    main()
