import math
import unittest

import polars as pl

from business_entity_resolution.feature_text import prepare_record, romanize_indic, skeleton
from business_entity_resolution.similarity_features import (
    CONTEXT_FEATURE_SCHEMA,
    PAIR_FEATURE_SCHEMA,
    add_context_features,
    calculate_pair_features,
    prepare_records,
)
from business_entity_resolution.token_stats import TokenStats, fit_token_stats


S1 = pl.DataFrame(
    {
        "entity_id": ["S1-1", "S1-2", "S1-3"],
        "business_name": ["Alpha Infra Private Limited", "Surgical Partners LLC", "Café Rivoli SARL"],
        "business_address": [
            "Cabin Number 5.1, 5Th Floor, Pitampura, Delhi",
            "865 Watson Road, Unit 108, Buckeye, AZ",
            "12 Rue de Rivoli, 75001 Paris",
        ],
        "country": ["India", "US", "France"],
    }
)
SECONDARY = pl.DataFrame(
    {
        "entity_id": ["S2-1", "S3-2", "S2-3", "S3-4"],
        "business_name": [
            "अल्फा इंफ्रा प्राइवेट लिमिटेड",
            "Fayepyralyra F/K/A Surgical Partners LLC",
            "Surgical Partners Inc",
            "CAFE RIVOLI",
        ],
        "business_address": [
            "CABIN NUMBER 5.1, 5TH FLOOR, PITAMPURA, DELHI, null, दिल्ली",
            "Unit 108, 865 Watson Road, AZ, Buckeye",
            "None",
            "PARIS, 12 RUE DE RIVOLI",
        ],
        "country": ["India", "US", "US", "France"],
    }
)
PAIRS = pl.DataFrame(
    {
        "source1_entity_id": ["S1-1", "S1-2", "S1-2", "S1-3", "S1-1"],
        "candidate_entity_id": ["S2-1", "S3-2", "S2-3", "S3-4", "S2-3"],
    }
)


def _features(pairs=PAIRS):
    return calculate_pair_features(pairs, S1, SECONDARY, TokenStats(), workers=1)


class FeatureTextTests(unittest.TestCase):
    def test_indic_romanisation_and_skeleton_align_with_latin(self):
        self.assertEqual(romanize_indic("भारत"), "bharat")
        hindi = prepare_record("अल्फा इंफ्रा प्राइवेट लिमिटेड", "")
        latin = prepare_record("Alpha Infra Pvt Ltd", "")
        self.assertEqual(hindi.name_core_skeleton, latin.name_core_skeleton)
        tamil = prepare_record("ஆல் கன்சல்டன்ட்ஸ் பிரைவேட் லிமிடெட்", "")
        self.assertEqual(tamil.name_core_skeleton, skeleton(("all", "consultants")))

    def test_unicode_view_is_preserved(self):
        record = prepare_record("अल्फा इंफ्रा", "")
        self.assertEqual(record.name_unicode, "अल्फा इंफ्रा")
        self.assertTrue(record.name_non_latin)

    def test_placeholder_addresses_are_missing(self):
        for value in ("", "None", "null", "N/A", " , null, "):
            self.assertTrue(prepare_record("X", value).address_missing, value)

    def test_region_names_collapse_to_codes(self):
        a = prepare_record("X", "5 Main St, Nisswa, Minnesota")
        b = prepare_record("X", "MN, NISSWA, 5 MAIN STREET")
        self.assertEqual(a.address_tokens, b.address_tokens)

    def test_alias_markers_split_names(self):
        record = prepare_record("Fayepyralyra F/K/A Surgical Partners LLC", "")
        self.assertEqual(record.name_alias_parts, ("fayepyralyra", "surgical partners"))


class PairFeatureTests(unittest.TestCase):
    def test_schema_order_and_dtypes(self):
        features = _features()
        self.assertEqual(features.height, PAIRS.height)
        feature_part = features.select(list(PAIR_FEATURE_SCHEMA))
        self.assertEqual(dict(feature_part.schema), PAIR_FEATURE_SCHEMA)
        for column in feature_part.columns:
            series = feature_part[column]
            if series.dtype.is_float():
                self.assertEqual(int(series.is_infinite().sum()), 0, column)

    def test_deterministic_and_order_preserving(self):
        first = _features()
        second = _features()
        self.assertTrue(first.equals(second))
        self.assertEqual(first["candidate_entity_id"].to_list(), PAIRS["candidate_entity_id"].to_list())

    def test_train_inference_parity_across_chunking_and_preparation(self):
        full = _features()
        prepared = calculate_pair_features(
            PAIRS.slice(2, 3), prepare_records(S1), prepare_records(SECONDARY), TokenStats(), workers=2
        )
        self.assertTrue(full.slice(2, 3).equals(prepared))

    def test_missing_address_is_nan_with_flags_not_zero(self):
        row = _features().filter(pl.col("candidate_entity_id") == "S2-3").row(0, named=True)
        self.assertEqual(row["addr_missing_b"], 1)
        self.assertEqual(row["addr_any_missing"], 1)
        for column in ("addr_ratio", "addr_token_set_ratio", "addr_idf_jaccard", "combo_min"):
            self.assertTrue(math.isnan(row[column]), column)

    def test_cross_script_and_unseen_country_pairs_score_high(self):
        features = _features()
        hindi = features.row(0, named=True)
        self.assertEqual(hindi["name_skeleton_token_set"], 1.0)
        self.assertEqual(hindi["name_script_mismatch"], 1)
        france = features.filter(pl.col("source1_entity_id") == "S1-3").row(0, named=True)
        self.assertEqual(france["same_country"], 1)
        self.assertEqual(france["name_exact_core"], 1)
        self.assertEqual(france["addr_token_set_ratio"], 1.0)

    def test_passthrough_columns_are_kept(self):
        pairs = PAIRS.with_columns(label=pl.lit(1, pl.Int8))
        self.assertIn("label", _features(pairs).columns)

    def test_unknown_record_raises(self):
        bad = pl.DataFrame({"source1_entity_id": ["S1-1"], "candidate_entity_id": ["S2-999"]})
        with self.assertRaises(KeyError):
            _features(bad)


class ContextAndStatsTests(unittest.TestCase):
    def test_context_features_schema_and_ranks(self):
        context = add_context_features(_features())
        for column, dtype in CONTEXT_FEATURE_SCHEMA.items():
            self.assertEqual(context.schema[column], dtype, column)
        s1_1 = context.filter(pl.col("source1_entity_id") == "S1-1").sort("ctx_combo_rank")
        self.assertEqual(s1_1["candidate_entity_id"][0], "S2-1")
        self.assertEqual(context.filter(pl.col("candidate_entity_id") == "S2-3")["ctx_rev_n_s1"][0], 2)

    def test_token_stats_roundtrip(self):
        import tempfile, pathlib

        stats = fit_token_stats([["acme", "corp"], ["acme"]], [["main"], ["main", "road"]], min_df=1)
        self.assertGreater(stats.name_idf("corp"), stats.name_idf("acme"))
        self.assertEqual(stats.name_idf("never-seen"), stats.name_idf("corp"))
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "stats.json"
            stats.save(path)
            loaded = TokenStats.load(path)
        self.assertEqual(loaded.name_df, stats.name_df)
        self.assertEqual(loaded.n_docs, stats.n_docs)


if __name__ == "__main__":
    unittest.main()
