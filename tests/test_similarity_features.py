import math
import unittest

import polars as pl

from business_entity_resolution.feature_text import prepare_record, romanize_indic, skeleton
from business_entity_resolution.similarity_features import (
    CONTEXT_FEATURE_SCHEMA,
    PAIR_FEATURE_SCHEMA,
    add_context_features,
    calculate_pair_features,
    calculate_pair_features_parallel,
    NA_SENTINEL,
    prepare_records,
    validate_feature_frame,
)
from business_entity_resolution.name_stats import (
    add_name_frequency_features, core_hash, load_token_noise, save_name_stats,
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
    return calculate_pair_features(pairs, S1, SECONDARY, TokenStats(), name_noise={}, workers=1)


def _write_name_stats(directory):
    from collections import Counter

    from business_entity_resolution.feature_text import prepare_name

    core = prepare_name("Fayepyralyra F/K/A Surgical Partners LLC")["name_core"]
    save_name_stats(
        directory, n_s1=10, n_secondary=10,
        core_s1=Counter({core_hash(core): 1}), core_secondary=Counter({core_hash(core): 3}),
        token_s1=Counter({"services": 1}), token_secondary=Counter({"services": 60}),
    )
    return directory


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
            PAIRS.slice(2, 3), prepare_records(S1), prepare_records(SECONDARY), TokenStats(), name_noise={}, workers=2
        )
        self.assertTrue(full.slice(2, 3).equals(prepared))

    def test_missing_address_is_sentinel_with_flags_not_zero(self):
        row = _features().filter(pl.col("candidate_entity_id") == "S2-3").row(0, named=True)
        self.assertEqual(row["addr_missing_b"], 1)
        self.assertEqual(row["addr_any_missing"], 1)
        for column in ("addr_ratio", "addr_token_set_ratio", "addr_idf_jaccard", "combo_min",
                       "addr_num_jaccard", "addr_primary_equal"):
            self.assertEqual(row[column], NA_SENTINEL, column)

    def test_no_nan_or_null_anywhere(self):
        features = _features()
        for column, dtype in PAIR_FEATURE_SCHEMA.items():
            self.assertEqual(features[column].null_count(), 0, column)
            if dtype.is_float():
                self.assertEqual(int(features[column].is_nan().sum()), 0, column)

    def test_validator_rejects_nan(self):
        import tempfile, pathlib

        with tempfile.TemporaryDirectory() as tmp:
            name_dir = _write_name_stats(pathlib.Path(tmp) / "names")
            good = add_name_frequency_features(_features(), name_dir)
        validate_feature_frame(good)
        bad = good.with_columns(pl.lit(float("nan"), pl.Float32).alias("addr_ratio"))
        with self.assertRaises(ValueError):
            validate_feature_frame(bad)

    def test_cross_script_and_unseen_country_pairs_score_high(self):
        features = _features()
        hindi = features.row(0, named=True)
        self.assertEqual(hindi["name_skeleton_token_set"], 1.0)
        self.assertEqual(hindi["name_script_mismatch"], 1)
        france = features.filter(pl.col("source1_entity_id") == "S1-3").row(0, named=True)
        self.assertEqual(france["same_country"], 1)
        self.assertEqual(france["name_exact_core"], 1)
        self.assertEqual(france["addr_token_set_ratio"], 1.0)

    def test_parallel_driver_matches_serial_exactly(self):
        import pathlib, tempfile

        pairs = PAIRS.sort("source1_entity_id")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "stats.json"
            TokenStats().save(path)
            name_dir = _write_name_stats(pathlib.Path(tmp) / "names")
            serial = add_name_frequency_features(
                calculate_pair_features(pairs, S1, SECONDARY, TokenStats(),
                                        name_noise=load_token_noise(name_dir), workers=1),
                name_dir,
            )
            parallel = calculate_pair_features_parallel(
                pairs, S1, SECONDARY, path, name_dir, processes=2, chunk_pairs=1
            )
        self.assertTrue(serial.equals(parallel))
        self.assertAlmostEqual(serial.filter(pl.col("candidate_entity_id") == "S3-2")["name_core_freq_sec_b"][0],
                               math.log1p(3), places=6)
        self.assertNotIn("_core_hash_a", serial.columns)

    def test_primary_number_separates_sibling_addresses(self):
        s1 = S1.with_columns(pl.lit("# 8-3-898/30/3, N Nagar Colony, Hyderabad").alias("business_address"))
        sec = SECONDARY.with_columns(pl.lit("DOOR NO 8-3-898/30/5, N NAGAR COLONY, HYDERABAD").alias("business_address"))
        pairs = pl.DataFrame({"source1_entity_id": ["S1-1"], "candidate_entity_id": ["S2-1"]})
        row = calculate_pair_features(pairs, s1, sec, TokenStats(), name_noise={}).row(0, named=True)
        self.assertEqual(row["addr_primary_equal"], 0.0)
        self.assertEqual(row["addr_num_only_a"], 0)  # "3" also appears elsewhere
        self.assertEqual(row["addr_num_only_b"], 1)
        self.assertGreater(row["addr_num_jaccard"], 0.7)

    def test_noise_ratio_of_unmatched_tokens(self):
        pairs = pl.DataFrame({"source1_entity_id": ["S1-2"], "candidate_entity_id": ["S2-3"]})
        sec = SECONDARY.with_columns(
            pl.when(pl.col("entity_id") == "S2-3").then(pl.lit("Surgical Partners Services"))
            .otherwise(pl.col("business_name")).alias("business_name"))
        row = calculate_pair_features(pairs, S1, sec, TokenStats(),
                                      name_noise={"services": 2.5}).row(0, named=True)
        self.assertEqual(row["name_b_unmatched_max_noise"], 2.5)
        self.assertEqual(row["name_a_unmatched_max_noise"], -5.0)  # all S1 tokens matched

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
