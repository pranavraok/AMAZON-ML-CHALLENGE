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


def _record(entity_id, name, address, country):
    return normalize_record(
        {
            "entity_id": entity_id,
            "business_name": name,
            "business_address": address,
            "country": country,
        }
    )


def test_exact_name_route_is_recorded_in_routes():
    """The exact-name route must be reachable, not silently skipped."""

    target = [_record("S2-001", "Alpha Beta Gamma", "1 Some Road", "India")]
    query = _record("S1-001", "Alpha Beta Gamma", "1 Some Road", "India")

    evidence = {
        c.candidate_entity_id: c
        for c in BlockingIndex(target).generate(query, "S2")
    }

    assert "exact_name" in evidence["S2-001"].routes
    assert "exact_address" in evidence["S2-001"].routes
    assert "exact_combined" in evidence["S2-001"].routes
    assert "token_sorted_name" in evidence["S2-001"].routes


def test_target_source_is_propagated_to_every_candidate():
    target = [_record("S2-001", "Alpha", "1 Road", "India")]
    query = _record("S1-001", "Alpha", "1 Road", "India")

    candidates = BlockingIndex(target).generate(query, "S2")

    assert candidates
    assert all(c.target_source == "S2" for c in candidates)


def test_address_token_pair_route_fires():
    """Two common address tokens co-occurring is usable evidence."""

    target = [
        _record("S2-001", "Zeta", "Flat 4 Bhandari Complex New Delhi", "India"),
        _record("S2-002", "Eta", "99 Somewhere Else entirely", "India"),
    ]
    query = _record(
        "S1-001",
        "Zeta",
        "Unit 7 Bhandari Complex New Delhi",
        "India",
    )

    evidence = {
        c.candidate_entity_id: c
        for c in BlockingIndex(target, address_pair_max_postings=5).generate(
            query, "S2"
        )
    }

    assert "S2-001" in evidence
    assert "address_token_pair" in evidence["S2-001"].routes


def test_address_token_pair_respects_posting_cap():
    target = [
        _record(f"S2-{i:03d}", f"Name {i}", "New Delhi", "India")
        for i in range(1, 21)
    ]
    query = _record("S1-001", "Name 1", "New Delhi", "India")

    evidence = {
        c.candidate_entity_id: c
        for c in BlockingIndex(target, address_pair_max_postings=5).generate(
            query, "S2"
        )
    }

    # ('delhi', 'new') now has 20 postings, above the cap of 5.
    assert all(
        "address_token_pair" not in c.routes for c in evidence.values()
    )


def test_blank_address_cannot_match_via_pair_route():
    target = [_record("S2-001", "Alpha", "", "India")]
    query = _record("S1-001", "Zeta", "", "India")

    evidence = {
        c.candidate_entity_id: c
        for c in BlockingIndex(target).generate(query, "S2")
    }

    assert "S2-001" not in evidence


def test_address_routes_do_not_cross_countries():
    target = [
        _record("S2-001", "Alpha", "12 MG Road", "India"),
        _record("S2-002", "Alpha", "12 MG Road", "France"),
    ]
    query = _record("S1-001", "Alpha", "12 MG Road", "France")

    evidence = {
        c.candidate_entity_id: c
        for c in BlockingIndex(target).generate(query, "S2")
    }

    assert "S2-002" in evidence
    assert "S2-001" not in evidence


def test_numeric_address_ignores_leading_zeros():
    target = [_record("S2-001", "Zeta", "F No 0014 A, New Delhi", "India")]
    query = _record("S1-001", "Zeta", "F No 14 A, New Delhi", "India")

    evidence = {
        c.candidate_entity_id: c
        for c in BlockingIndex(target).generate(query, "S2")
    }

    assert "numeric_address" in evidence["S2-001"].routes


def test_index_rejects_records_missing_normalized_fields():
    class Bad:
        entity_id = "S2-001"
        country = "India"

    try:
        BlockingIndex([Bad()])
    except ValueError as error:
        assert "name_unicode" in str(error)
    else:
        raise AssertionError("expected ValueError for non-normalized record")