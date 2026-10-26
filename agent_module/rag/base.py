"""База литературных аргументов: загрузка из CSV.

Формат ``data/arguments.csv`` (UTF-8, разделитель — запятая):
    id        уникальный идентификатор (a001, a002, ...)
    title     произведение
    author    автор
    argument  краткий пересказ эпизода и вывод, который можно использовать как аргумент
    tags      темы через «;» (3–5 штук из словаря agent_module/rag/tags.py)
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_ARGUMENTS_PATH = DATA_DIR / "arguments.csv"
TAG_SEPARATOR = ";"


@dataclass(frozen=True)
class Argument:
    id: str
    title: str
    author: str
    argument: str
    tags: List[str] = field(default_factory=list)

    @property
    def passage(self) -> str:
        """Текст, который индексируется эмбеддером."""
        return f"{self.title} — {self.author}. {self.argument}"


def split_tags(raw: str) -> List[str]:
    return [t.strip().lower() for t in (raw or "").split(TAG_SEPARATOR) if t.strip()]


def load_arguments(path: Optional[Union[str, Path]] = None) -> List[Argument]:
    csv_path = Path(path) if path else DEFAULT_ARGUMENTS_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"База аргументов не найдена: {csv_path}")
    items: List[Argument] = []
    seen = set()
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            arg_id = (row.get("id") or "").strip()
            if not arg_id:
                raise ValueError(f"{csv_path}: строка без id: {row}")
            if arg_id in seen:
                raise ValueError(f"{csv_path}: повторяющийся id {arg_id}")
            seen.add(arg_id)
            items.append(
                Argument(
                    id=arg_id,
                    title=(row.get("title") or "").strip(),
                    author=(row.get("author") or "").strip(),
                    argument=(row.get("argument") or "").strip(),
                    tags=split_tags(row.get("tags", "")),
                )
            )
    if not items:
        raise ValueError(f"{csv_path}: база пуста")
    return items
