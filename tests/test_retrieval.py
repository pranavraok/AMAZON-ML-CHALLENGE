from business_entity_resolution.normalization import normalize_record
from business_entity_resolution.retrieval import (
    build_address_retriever,
    build_name_retriever,
)


def test_name_tfidf_retrieves_similar_name():
    target_records = [
        normalize_record(
            {
                "entity_id": "S2-001",
                "business_name": "Amazon Technologies Private Limited",
                "business_address": "12 MG Road Bangalore",
                "country": "India",
            }
        ),
        normalize_record(
            {
                "entity_id": "S2-002",
                "business_name": "Completely Different Business",
                "business_address": "50 Main Road Delhi",
                "country": "India",
            }
        ),
    ]

    source1 = normalize_record(
        {
            "entity_id": "S1-001",
            "business_name": "Amazon Technologies Pvt Limited",
            "business_address": "12 MG Road Bangalore",
            "country": "India",
        }
    )

    retriever = build_name_retriever(
        target_records,
        top_k=2,
    )

    results = retriever.retrieve(source1)

    candidate_ids = {
        result.candidate_entity_id
        for result in results
    }

    assert "S2-001" in candidate_ids


def test_address_tfidf_retrieves_similar_address():
    target_records = [
        normalize_record(
            {
                "entity_id": "S2-001",
                "business_name": "ABC",
                "business_address": "12 Mahatma Gandhi Road Bangalore",
                "country": "India",
            }
        ),
        normalize_record(
            {
                "entity_id": "S2-002",
                "business_name": "XYZ",
                "business_address": "500 Delhi Main Street",
                "country": "India",
            }
        ),
    ]

    source1 = normalize_record(
        {
            "entity_id": "S1-001",
            "business_name": "Something",
            "business_address": "12 MG Road Bangalore",
            "country": "India",
        }
    )

    retriever = build_address_retriever(
        target_records,
        top_k=2,
    )

    results = retriever.retrieve(source1)

    candidate_ids = {
        result.candidate_entity_id
        for result in results
    }

    assert "S2-001" in candidate_ids


def test_blank_address_returns_no_results():
    target_records = [
        normalize_record(
            {
                "entity_id": "S2-001",
                "business_name": "ABC",
                "business_address": "",
                "country": "India",
            }
        )
    ]

    source1 = normalize_record(
        {
            "entity_id": "S1-001",
            "business_name": "ABC",
            "business_address": "",
            "country": "India",
        }
    )

    retriever = build_address_retriever(target_records)

    results = retriever.retrieve(source1)

    assert results == []


def test_country_partition_is_respected():
    target_records = [
        normalize_record(
            {
                "entity_id": "S2-001",
                "business_name": "ABC Technologies",
                "business_address": "Main Road",
                "country": "India",
            }
        ),
        normalize_record(
            {
                "entity_id": "S2-002",
                "business_name": "ABC Technologies",
                "business_address": "Main Road",
                "country": "France",
            }
        ),
    ]

    source1 = normalize_record(
        {
            "entity_id": "S1-001",
            "business_name": "ABC Technologies",
            "business_address": "Main Road",
            "country": "France",
        }
    )

    retriever = build_name_retriever(
        target_records,
        top_k=10,
    )

    results = retriever.retrieve(source1)

    candidate_ids = {
        result.candidate_entity_id
        for result in results
    }

    assert "S2-002" in candidate_ids
    assert "S2-001" not in candidate_ids