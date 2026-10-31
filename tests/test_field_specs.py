import unittest

from agents.field_specs import (
    FIELD_SPECS,
    OFFICIAL_FIELD_LABELS,
    get_field_spec,
    get_field_specs,
)


EXPECTED_LABELS = (
    "Bid Number",
    "Title",
    "Due Date",
    "Bid Submission Type",
    "Term of Bid",
    "Pre Bid Meeting",
    "Installation",
    "Bid Bond Requirement",
    "Delivery Date",
    "Payment Terms",
    "Any Additional Documentation Required",
    "MFG for Registration",
    "Contract or Cooperative to use",
    "Model_no",
    "Part_no",
    "Product",
    "contact_info",
    "company_name",
    "Bid Summary",
    "Product Specification",
)


class FieldRegistryTests(unittest.TestCase):
    def test_exact_labels_spelling_order_and_count(self):
        self.assertEqual(OFFICIAL_FIELD_LABELS, EXPECTED_LABELS)
        self.assertEqual(
            tuple(spec.field_name for spec in FIELD_SPECS),
            EXPECTED_LABELS,
        )
        self.assertEqual(len(FIELD_SPECS), 20)
        self.assertEqual(len(set(OFFICIAL_FIELD_LABELS)), 20)

    def test_every_spec_is_complete_and_bid_agnostic(self):
        serialized = " ".join(
            spec.model_dump_json() for spec in FIELD_SPECS
        )
        self.assertNotIn("Bid1", serialized)
        self.assertNotIn("Bid2", serialized)

        for spec in FIELD_SPECS:
            self.assertTrue(spec.questions)
            self.assertTrue(all(question.strip() for question in spec.questions))
            self.assertTrue(spec.expected_type)
            self.assertTrue(spec.specialist_group)
            self.assertIn("preferred_doc_types", spec.search_filters)

    def test_registry_accessors(self):
        self.assertIs(get_field_specs(), FIELD_SPECS)
        self.assertEqual(
            get_field_spec("Product Specification").field_name,
            "Product Specification",
        )
        with self.assertRaises(KeyError):
            get_field_spec("Unsupported Field")


if __name__ == "__main__":
    unittest.main()
