"""Policy tests for the PR eval bot — the third GPU-free file in tests/.

The bot spends real money and posts public verdicts, so every decision it
makes (who is eligible, what counts as evaluated, how a verdict is derived
from measurements) is a pure function tested here without a GPU, a box, or a
GitHub token. The shell/ssh plumbing is deliberately not mocked-and-tested:
its correctness is established by the live smoke run, and a mock of rsync
would only test the mock.

Runs under pytest and as `python3 tests/test_pr_eval_bot.py` (CI does the
latter — the runner has no pytest, and the GPU suite gate runs it via pytest
anyway).
"""
import importlib.util
import pathlib
import unittest

_BOT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "pr_eval_bot.py"
_spec = importlib.util.spec_from_file_location("pr_eval_bot", _BOT)
bot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bot)


class TickedCheckbox(unittest.TestCase):
    def test_ticked(self):
        self.assertTrue(bot.ticked_5090("x\n- [x] Tested on **RTX 5090** — done\ny"))

    def test_capital_x_and_spaces(self):
        self.assertTrue(bot.ticked_5090("- [ X ] Tested on RTX 5090"))

    def test_unticked(self):
        self.assertFalse(bot.ticked_5090("- [ ] Tested on **RTX 5090**"))

    def test_tick_on_a_non_5090_line_does_not_count(self):
        self.assertFalse(bot.ticked_5090("- [x] Parity test added\n- [ ] RTX 5090"))

    def test_no_body(self):
        self.assertFalse(bot.ticked_5090(None))
        self.assertFalse(bot.ticked_5090(""))


class RuntimePaths(unittest.TestCase):
    def test_engine_and_tests_count(self):
        self.assertTrue(bot.touches_runtime(["braid/model/gdn.py"]))
        self.assertTrue(bot.touches_runtime(["README.md", "tests/test_x.py"]))

    def test_docs_and_scripts_do_not(self):
        self.assertFalse(bot.touches_runtime(["docs/ROADMAP.md", "scripts/x.sh"]))

    def test_prefix_is_a_directory_not_a_substring(self):
        self.assertFalse(bot.touches_runtime(["braided/other.py"]))


class Idempotency(unittest.TestCase):
    SHA = "abc123def456"

    def test_verdict_parks_the_head(self):
        bodies = ["hi", bot.marker(self.SHA) + "\nverdict"]
        self.assertTrue(bot.already_evaluated(bodies, self.SHA))

    def test_other_heads_are_not_parked(self):
        bodies = [bot.marker("othersha") + "\nverdict"]
        self.assertFalse(bot.already_evaluated(bodies, self.SHA))

    def test_one_error_gets_a_retry_two_do_not(self):
        one = [bot.error_marker(self.SHA)]
        self.assertFalse(bot.already_evaluated(one, self.SHA))
        two = one + ["x", bot.error_marker(self.SHA)]
        self.assertTrue(bot.already_evaluated(two, self.SHA))

    def test_none_bodies_are_tolerated(self):
        self.assertFalse(bot.already_evaluated([None, ""], self.SHA))


class BenchParsing(unittest.TestCase):
    OUT = (
        "host health: 2812 MHz SM ...\n"
        '{"unrelated": 1}\n'
        '{"arms": [{"name": "eager-torch", "batch": 16, "tok_per_s": 100.0},'
        ' {"name": "graphed-kvbucket", "batch": 16, "tok_per_s": 811.5},'
        ' {"name": "graphed-kvbucket", "batch": 64, "tok_per_s": 5100.2}],'
        ' "health": "ok"}\n'
    )

    def test_picks_arm_and_batch(self):
        self.assertEqual(bot.pick_tok_s(self.OUT, 16), 811.5)
        self.assertEqual(bot.pick_tok_s(self.OUT, 64), 5100.2)

    def test_missing_arm_is_an_error_not_a_zero(self):
        with self.assertRaises(ValueError):
            bot.pick_tok_s(self.OUT, 128)

    def test_no_json_is_an_error(self):
        with self.assertRaises(ValueError):
            bot.pick_tok_s("OOM at batch 64: ...\n", 16)


class Verdict(unittest.TestCase):
    def test_suite_failure_rejects_regardless_of_numbers(self):
        label, _ = bot.verdict(False, {16: 50.0})
        self.assertEqual(label, "eval:reject")

    def test_regression_outranks_improvement(self):
        label, reason = bot.verdict(True, {16: 8.0, 64: -3.0})
        self.assertEqual(label, "eval:reject")
        self.assertIn("B=64", reason)

    def test_gain_beyond_the_bar_passes(self):
        label, reason = bot.verdict(True, {16: 0.4, 64: 5.2})
        self.assertEqual(label, "eval:pass")
        self.assertIn("B=64", reason)

    def test_within_the_bar_is_noise_not_pass(self):
        label, _ = bot.verdict(True, {16: 1.9, 64: -1.9})
        self.assertEqual(label, "eval:noise")

    def test_the_bar_itself_is_noise(self):
        label, _ = bot.verdict(True, {16: 2.0, 64: -2.0})
        self.assertEqual(label, "eval:noise")


class Eligibility(unittest.TestCase):
    def info(self, **kw):
        base = {
            "state": "OPEN", "isDraft": False, "labels": [],
            "headRefOid": "feedbeef1234",
            "body": "- [x] Tested on **RTX 5090**",
            "files": [{"path": "braid/model/gdn.py"}], "comments": [],
        }
        base.update(kw)
        return base

    def test_happy_path(self):
        ok, why = bot.eligible(self.info())
        self.assertTrue(ok, why)

    def test_draft_and_hold_are_skipped(self):
        self.assertFalse(bot.eligible(self.info(isDraft=True))[0])
        self.assertFalse(bot.eligible(self.info(labels=[{"name": "hold"}]))[0])

    def test_docs_only_is_skipped(self):
        ok, _ = bot.eligible(self.info(files=[{"path": "README.md"}]))
        self.assertFalse(ok)

    def test_unticked_needs_the_eval_label(self):
        no_tick = self.info(body="- [ ] Tested on **RTX 5090**")
        self.assertFalse(bot.eligible(no_tick)[0])
        forced = self.info(body="- [ ] Tested on **RTX 5090**",
                           labels=[{"name": "eval"}])
        self.assertTrue(bot.eligible(forced)[0])

    def test_evaluated_head_is_parked(self):
        done = self.info(comments=[{"body": bot.marker("feedbeef1234")}])
        self.assertFalse(bot.eligible(done)[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
