from business_entity_resolution.blocking import BlockingIndex
from business_entity_resolution.normalization import normalize_record


def test_exact_name_blocking():
    target_records = [
        normalize_record(
            {
                "entity_id": "S2-001",
                "business_name": "ABC Technologies Pvt Ltd",
                "business_address": "12 MG Road Bangalore",
                "country": "India",
            }
        ),
        normalize_record(
            {
                "entity_id": "S2-002",
                "business_name": "Different Business",
                "business_address": "50 Main Road",
                "country": "India",
            }
        ),
    ]

    source1 = normalize_record(
        {
            "entity_id": "S1-001",
            "business_name": "ABC Technologies Pvt Ltd",
            "business_address": "12 MG Road Bangalore",
            "country": "India",
        }
    )

    index = BlockingIndex(target_records)

    candidates = index.generate(source1)

    candidate_ids = {
        candidate.candidate_entity_id
        for candidate in candidates
    }

    assert "S2-001" in candidate_ids
    assert "S2-002" not in candidate_ids


def test_token_sorted_name_blocking():
    target_records = [
        normalize_record(
            {
                "entity_id": "S2-001",
                "business_name": "Technologies ABC",
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

    index = BlockingIndex(target_records)

    candidates = index.generate(source1)

    candidate_ids = {
        candidate.candidate_entity_id
        for candidate in candidates
    }

    assert "S2-001" in candidate_ids


def test_blank_addresses_do_not_match():
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
            "business_name": "XYZ",
            "business_address": "",
            "country": "India",
        }
    )

    index = BlockingIndex(target_records)

    candidates = index.generate(source1)

    candidate_ids = {
        candidate.candidate_entity_id
        for candidate in candidates
    }

    assert "S2-001" not in candidate_ids