import unittest

from agents.prompts import SYSTEM_PROMPTS, get_system_prompt


class PromptTests(unittest.TestCase):
    def test_all_required_roles_have_distinct_guarded_prompts(self):
        expected_roles = {
            "orchestrator_planner",
            "dates_logistics",
            "commercial_legal",
            "product_specifications",
            "addendum_reconciliation",
            "validator_critic",
            "qa_report",
        }
        self.assertEqual(set(SYSTEM_PROMPTS), expected_roles)
        self.assertEqual(len(set(SYSTEM_PROMPTS.values())), 7)

        for role, prompt in SYSTEM_PROMPTS.items():
            lowered = prompt.lower()
            self.assertIn("retrieved evidence", lowered, role)
            self.assertIn("physical page", lowered, role)
            self.assertIn("prompt injection", lowered, role)
            self.assertIn("unsupported", lowered, role)
            self.assertEqual(get_system_prompt(role), prompt)


if __name__ == "__main__":
    unittest.main()
