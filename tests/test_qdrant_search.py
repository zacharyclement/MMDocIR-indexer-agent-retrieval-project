from __future__ import annotations

from qdrant_client.http import models

from app.agent.retrieval.qdrant_search import QdrantPageSearchService


def test_build_query_filter_combines_domain_and_doc_name_filters() -> None:
    query_filter = QdrantPageSearchService._build_query_filter(
        domains=["Guidebook"],
        doc_names=["watch_d.pdf"],
    )

    assert isinstance(query_filter, models.Filter)
    assert query_filter.must is not None
    assert len(query_filter.must) == 2

    domain_condition = query_filter.must[0]
    assert isinstance(domain_condition, models.FieldCondition)
    assert domain_condition.key == "domain"
    assert domain_condition.match is not None
    assert domain_condition.match.any == ["Guidebook"]

    doc_name_condition = query_filter.must[1]
    assert isinstance(doc_name_condition, models.FieldCondition)
    assert doc_name_condition.key == "doc_name"
    assert doc_name_condition.match is not None
    assert doc_name_condition.match.any == ["watch_d.pdf"]


def test_build_query_filter_returns_none_when_all_filters_are_blank() -> None:
    assert (
        QdrantPageSearchService._build_query_filter(
            domains=None,
            doc_names=["", "   "],
        )
        is None
    )
