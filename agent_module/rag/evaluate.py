"""Оценка поиска: Hit Rate@k — доля запросов, у которых среди первых k результатов есть хотя бы один релевантный аргумент."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from agent_module.rag.base import DATA_DIR
from agent_module.rag.retriever import MODE_HYBRID, MODE_VECTOR, HybridRetriever

DEFAULT_EVAL_PATH = DATA_DIR / "rag_eval.csv"


@dataclass(frozen=True)
class EvalQuery:
    query: str
    relevant_ids: frozenset


def load_eval_queries(path: Optional[Union[str, Path]] = None) -> List[EvalQuery]:
    csv_path = Path(path) if path else DEFAULT_EVAL_PATH
    queries: List[EvalQuery] = []
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            relevant = frozenset(i.strip() for i in row["relevant_ids"].split(";") if i.strip())
            queries.append(EvalQuery(row["query"].strip(), relevant))
    return queries


def hit_rates(
    retriever: HybridRetriever,
    queries: Sequence[EvalQuery],
    mode: str,
    ks: Sequence[int] = (1, 3),
) -> Tuple[Dict[int, float], List[dict]]:
    """Возвращает {k: Hit Rate@k} и разбор по каждому запросу."""
    max_k = max(ks)
    hits_at = {k: 0 for k in ks}
    details: List[dict] = []
    for item in queries:
        outcome = retriever.search(item.query, k=max_k, mode=mode)
        ranked = [hit.argument.id for hit in outcome.hits]
        row = {
            "query": item.query,
            "top": ranked,
            "query_tags": outcome.query_tags,
            "tag_filter_applied": outcome.tag_filter_applied,
        }
        for k in ks:
            ok = any(i in item.relevant_ids for i in ranked[:k])
            hits_at[k] += int(ok)
            row[f"hit@{k}"] = ok
        details.append(row)
    total = max(len(queries), 1)
    return {k: hits_at[k] / total for k in ks}, details


def evaluate(
    retriever: HybridRetriever,
    queries: Sequence[EvalQuery],
    modes: Sequence[str] = (MODE_VECTOR, MODE_HYBRID),
    ks: Sequence[int] = (1, 3),
) -> Dict[str, dict]:
    report: Dict[str, dict] = {}
    for mode in modes:
        rates, details = hit_rates(retriever, queries, mode, ks)
        report[mode] = {"hit_rate": {f"@{k}": round(v, 4) for k, v in rates.items()}, "details": details}
    return report
