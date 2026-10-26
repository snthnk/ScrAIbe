import csv

import pytest

from agent_module.models import RagResult
from agent_module.rag import HybridRetriever, load_arguments
from agent_module.rag.base import DEFAULT_ARGUMENTS_PATH
from agent_module.rag.embedder import HashingEmbedder
from agent_module.rag.evaluate import hit_rates, load_eval_queries
from agent_module.rag.retriever import MODE_HYBRID, MODE_VECTOR
from agent_module.rag.tags import TAG_KEYWORDS, extract_tags


@pytest.fixture(scope="module")
def retriever():
    return HybridRetriever.build(load_arguments(), HashingEmbedder())


def test_base_is_consistent():
    items = load_arguments()
    assert len(items) >= 150
    assert len({a.id for a in items}) == len(items)
    for item in items:
        assert item.title and item.author and len(item.argument) > 40, item.id
        assert 3 <= len(item.tags) <= 5, item.id
        assert set(item.tags) <= set(TAG_KEYWORDS), (item.id, set(item.tags) - set(TAG_KEYWORDS))


def test_load_arguments_rejects_duplicates(tmp_path):
    path = tmp_path / "bad.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "title", "author", "argument", "tags"])
        w.writerow(["a1", "T", "A", "text", "война"])
        w.writerow(["a1", "T2", "A2", "text", "война"])
    with pytest.raises(ValueError, match="повторяющийся"):
        load_arguments(path)


def test_eval_queries_reference_existing_arguments():
    ids = {a.id for a in load_arguments()}
    queries = load_eval_queries()
    assert len(queries) >= 80
    for q in queries:
        assert q.relevant_ids and q.relevant_ids <= ids, q.query


def test_extract_tags_uses_lemmas():
    assert "война" in extract_tags("Почему важно помнить о войнах прошлого?")
    assert "природа" in extract_tags("Как лес и реки влияют на человека")
    assert extract_tags("Привет") == []


def test_search_returns_k_hits_sorted_by_score(retriever):
    outcome = retriever.search("роль книги в жизни человека", k=3, mode=MODE_VECTOR)
    assert len(outcome.hits) == 3
    scores = [h.score for h in outcome.hits]
    assert scores == sorted(scores, reverse=True)
    assert not outcome.tag_filter_applied


def test_exact_passage_is_found_first(retriever):
    target = retriever.arguments[10]
    outcome = retriever.search(target.passage, k=1, mode=MODE_VECTOR)
    assert outcome.hits[0].argument.id == target.id


def test_hybrid_keeps_only_tag_matches_when_enough(retriever):
    outcome = retriever.search("что угодно про войну", k=3, mode=MODE_HYBRID, tags=["война"])
    assert outcome.tag_filter_applied
    assert outcome.query_tags == ["война"]
    assert all("война" in h.argument.tags for h in outcome.hits)
    assert all(h.matched_tags == ["война"] for h in outcome.hits)


def test_hybrid_backfills_when_few_tag_matches(retriever):
    rare = [t for t in TAG_KEYWORDS if sum(t in a.tags for a in retriever.arguments) == 1][:1]
    if not rare:
        pytest.skip("в базе нет темы с единственным аргументом")
    outcome = retriever.search("пример для сочинения", k=3, mode=MODE_HYBRID, tags=rare)
    assert len(outcome.hits) == 3
    assert rare[0] in outcome.hits[0].argument.tags


def test_hybrid_without_query_tags_equals_vector(retriever):
    query = "zzzz qqqq"
    assert extract_tags(query) == []
    vec = retriever.search(query, k=3, mode=MODE_VECTOR)
    hyb = retriever.search(query, k=3, mode=MODE_HYBRID)
    assert [h.argument.id for h in vec.hits] == [h.argument.id for h in hyb.hits]
    assert not hyb.tag_filter_applied


def test_empty_query_and_bad_mode(retriever):
    assert retriever.search("   ").hits == []
    with pytest.raises(ValueError):
        retriever.search("x", mode="magic")


def test_index_cache_roundtrip(tmp_path):
    embedder = HashingEmbedder(dim=256)
    first = HybridRetriever.from_csv(DEFAULT_ARGUMENTS_PATH, embedder, index_dir=tmp_path)
    assert (tmp_path / "faiss.index").exists() and (tmp_path / "meta.json").exists()
    second = HybridRetriever.from_csv(DEFAULT_ARGUMENTS_PATH, embedder, index_dir=tmp_path)
    q = "дружба и верность"
    assert [h.argument.id for h in first.search(q).hits] == [h.argument.id for h in second.search(q).hits]


def test_index_is_rebuilt_when_embedder_changes(tmp_path):
    HybridRetriever.from_csv(DEFAULT_ARGUMENTS_PATH, HashingEmbedder(dim=256), index_dir=tmp_path)
    other = HybridRetriever.from_csv(DEFAULT_ARGUMENTS_PATH, HashingEmbedder(dim=128), index_dir=tmp_path)
    assert other.index.d == 128


def test_hit_rate_math_on_toy_queries(retriever):
    from agent_module.rag.evaluate import EvalQuery

    target = retriever.arguments[5]
    queries = [EvalQuery(target.passage, frozenset({target.id})), EvalQuery(target.passage, frozenset({"нет-такого"}))]
    rates, details = hit_rates(retriever, queries, MODE_VECTOR, ks=(1, 3))
    assert rates == {1: 0.5, 3: 0.5}
    assert details[0]["hit@1"] is True and details[1]["hit@3"] is False


def test_literature_rag_tool_returns_valid_result(monkeypatch, retriever):
    from agent_module.tools import rag as rag_tool

    monkeypatch.setattr(rag_tool, "get_retriever", lambda: retriever)
    raw = rag_tool.literature_rag.invoke({"query": "память о войне"})
    result = RagResult.model_validate(raw)
    assert result.status == "ok"
    assert 1 <= len(result.results) <= 3
    assert result.results[0].title and result.results[0].argument
    assert result.retrieval["mode"] == MODE_HYBRID


def test_literature_rag_tool_reports_unavailable_instead_of_crashing(monkeypatch):
    from agent_module.tools import rag as rag_tool

    def boom():
        raise RuntimeError("модель эмбеддингов не скачана")

    monkeypatch.setattr(rag_tool, "get_retriever", boom)
    result = RagResult.model_validate(rag_tool.literature_rag.invoke({"query": "любовь"}))
    assert result.status == "unavailable"
    assert "не скачана" in result.message
