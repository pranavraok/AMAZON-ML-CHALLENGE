import unittest

from business_entity_resolution.normalization import (
    fold_ascii,
    normalize_record,
    normalize_unicode,
)


class NormalizationTests(unittest.TestCase):
    def test_case_punctuation_ampersand_and_accents(self) -> None:
        self.assertEqual(
            normalize_unicode("  Caf\u00e9 & Sons, Pvt. Ltd. "),
            "caf\u00e9 and sons pvt ltd",
        )
        self.assertEqual(
            fold_ascii("Caf\u00e9 & Sons"),
            "cafe and sons",
        )

    def test_unicode_view_preserves_local_script(self) -> None:
        value = "\u09aa\u09b6\u09cd\u099a\u09bf\u09ae\u09ac\u0999\u09cd\u0997"
        self.assertEqual(normalize_unicode(value), value)
        self.assertEqual(fold_ascii(value), "")

    def test_normalized_record_contract(self) -> None:
        normalized = normalize_record(
            {
                "entity_id": "S1-1",
                "business_name": "ACME, Inc.",
                "business_address": "12-B Main Road, Pune 411001",
                "country": "India",
            }
        )
        self.assertEqual(normalized.entity_id, "S1-1")
        self.assertEqual(normalized.name_unicode, "acme inc")
        self.assertEqual(normalized.name_token_sorted, "acme inc")
        self.assertEqual(normalized.address_numbers, ("12", "411001"))
        self.assertFalse(normalized.address_missing)

    def test_missing_address_is_explicit(self) -> None:
        normalized = normalize_record(
            {
                "entity_id": "S2-1",
                "business_name": "Example",
                "business_address": "",
                "country": "US",
            }
        )
        self.assertTrue(normalized.address_missing)
        self.assertEqual(normalized.address_tokens, ())


if __name__ == "__main__":
    unittest.main()
