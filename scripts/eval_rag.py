"""Оценка RAG: Hit Rate@1 / @3 для чистого векторного поиска и для гибридного (вектор + фильтр по тегам).

    python scripts/eval_rag.py                       # боевой режим: multilingual-e5-large
    python scripts/eval_rag.py --embedder hash       # офлайн-проверка конвейера (без скачивания модели)
    python scripts/eval_rag.py --model intfloat/multilingual-e5-small

Результат печатается в консоль и сохраняется в results/rag_eval_<эмбеддер>.json и .md.
Эмбеддер ``hash`` нужен только чтобы проверить код без модели — его числа не отражают качество поиска.
"""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent_module.rag.base import DEFAULT_ARGUMENTS_PATH, load_arguments  # noqa: E402
from agent_module.rag.embedder import build_embedder  # noqa: E402
from agent_module.rag.evaluate import DEFAULT_EVAL_PATH, evaluate, load_eval_queries  # noqa: E402
from agent_module.rag.retriever import MODE_HYBRID, MODE_VECTOR, HybridRetriever  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedder", default="e5", choices=["e5", "hash"])
    parser.add_argument("--model", help="имя модели sentence-transformers (по умолчанию intfloat/multilingual-e5-large)")
    parser.add_argument("--arguments", default=str(DEFAULT_ARGUMENTS_PATH))
    parser.add_argument("--queries", default=str(DEFAULT_EVAL_PATH))
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    parser.add_argument("--show-misses", action="store_true", help="печатать запросы, не попавшие в top-3")
    args = parser.parse_args()

    if args.model:
        os.environ["RAG_EMBEDDER_MODEL"] = args.model
    embedder = build_embedder(args.embedder)
    arguments = load_arguments(args.arguments)
    queries = load_eval_queries(args.queries)
    retriever = HybridRetriever.build(arguments, embedder)
    report = evaluate(retriever, queries, modes=(MODE_VECTOR, MODE_HYBRID))

    print(f"Эмбеддер: {embedder.name} | аргументов в базе: {len(arguments)} | запросов: {len(queries)}")
    lines = ["| Режим | Hit Rate@1 | Hit Rate@3 |", "|---|---|---|"]
    for mode, data in report.items():
        rate = data["hit_rate"]
        lines.append(f"| {mode} | {rate['@1']:.2f} | {rate['@3']:.2f} |")
    table = "\n".join(lines)
    print(table)

    if args.show_misses:
        for mode, data in report.items():
            print(f"\nПромахи @3 ({mode}):")
            for row in data["details"]:
                if not row["hit@3"]:
                    print(f"  - {row['query']} -> {row['top']}  (теги запроса: {row['query_tags']})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"rag_eval_{embedder.name.replace('/', '_')}"
    payload = {"embedder": embedder.name, "arguments": len(arguments), "queries": len(queries), "report": report}
    (out_dir / f"{stem}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    note = "" if args.embedder != "hash" else "\n\n> Эмбеддер `hash` — офлайн-заглушка для проверки конвейера, не оценка качества.\n"
    (out_dir / f"{stem}.md").write_text(
        f"# Оценка RAG\n\nЭмбеддер: `{embedder.name}`; аргументов: {len(arguments)}; запросов: {len(queries)}\n\n{table}{note}",
        encoding="utf-8",
    )
    print(f"\nСохранено: {out_dir / (stem + '.json')}")


if __name__ == "__main__":
    main()
