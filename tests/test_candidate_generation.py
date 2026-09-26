from business_entity_resolution.candidate_generation import (
    CandidateGenerationConfig,
    CandidateGenerator,
)
from business_entity_resolution.normalization import normalize_record


def test_candidate_generator_combines_blocking_and_tfidf():

    target_records = [
        normalize_record(
            {
                "entity_id": "S2-001",
                "business_name": "ABC Technologies Private Limited",
                "business_address": "12 MG Road Bangalore",
                "country": "India",
            }
        ),
        normalize_record(
            {
                "entity_id": "S2-002",
                "business_name": "Completely Different Company",
                "business_address": "500 Delhi Road",
                "country": "India",
            }
        ),
        normalize_record(
            {
                "entity_id": "S2-003",
                "business_name": "Technologies ABC",
                "business_address": "12 MG Road Bangalore",
                "country": "India",
            }
        ),
    ]

    source1 = normalize_record(
        {
            "entity_id": "S1-001",
            "business_name": "ABC Technologies Pvt Limited",
            "business_address": "12 MG Road Bangalore",
            "country": "India",
        }
    )

    generator = CandidateGenerator(
        target_records,
        config=CandidateGenerationConfig(
            name_top_k=3,
            address_top_k=3,
        ),
    )

    candidates = generator.generate_for_one(source1)

    candidate_ids = {
        candidate.candidate_entity_id
        for candidate in candidates
    }

    assert "S2-001" in candidate_ids
    assert "S2-003" in candidate_ids

    # No duplicate candidate IDs.
    assert len(candidate_ids) == len(candidates)


def test_candidate_metadata_contains_routes():

    target_records = [
        normalize_record(
            {
                "entity_id": "S2-001",
                "business_name": "ABC Technologies",
                "business_address": "12 MG Road",
                "country": "India",
            }
        )
    ]

    source1 = normalize_record(
        {
            "entity_id": "S1-001",
            "business_name": "ABC Technologies",
            "business_address": "12 MG Road",
            "country": "India",
        }
    )

    generator = CandidateGenerator(target_records)

    candidates = generator.generate_for_one(source1)

    candidate = next(
        candidate
        for candidate in candidates
        if candidate.candidate_entity_id == "S2-001"
    )

    assert "exact_name" in candidate.routes
    assert "name_tfidf" in candidate.routes
    assert "exact_address" in candidate.routes
    assert candidate.retrieval_score > 0


def test_country_partition_is_preserved():

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

    generator = CandidateGenerator(target_records)

    candidates = generator.generate_for_one(source1)

    candidate_ids = {
        candidate.candidate_entity_id
        for candidate in candidates
    }

    assert "S2-002" in candidate_ids
    assert "S2-001" not in candidate_ids