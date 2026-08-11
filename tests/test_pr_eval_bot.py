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
import json
import os
import pathlib
import tempfile
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

    def test_a_verdict_from_an_older_policy_does_not_park_the_head(self):
        """The staleness mechanism. PR #5 was stranded because a verdict
        reached under the old policy parked its head forever, so widening
        the policy could never reach it. Bumping POLICY_VERSION must make
        that verdict stop counting, so the head is measured again under the
        rules now in force rather than settled on a stale label."""
        stale = f"<!-- braid-eval sha={self.SHA} policy={bot.POLICY_VERSION - 1} -->"
        self.assertFalse(bot.already_evaluated([stale + "\nverdict"], self.SHA))
        self.assertTrue(bot.already_evaluated([bot.marker(self.SHA)], self.SHA))

    def test_the_marker_carries_the_policy_version(self):
        self.assertIn(f"policy={bot.POLICY_VERSION}", bot.marker(self.SHA))

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
        # a small regression alongside a gain lands in the slower band,
        # a material one is a reject — either way the gain never wins
        label, reason = bot.verdict(True, {16: 8.0, 64: -3.0})
        self.assertEqual(label, "eval:slower")
        self.assertIn("B=64", reason)
        label, reason = bot.verdict(True, {16: 8.0, 64: -9.0})
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


class GradedTiers(unittest.TestCase):
    """The ladder: the tier scales the evidence, and the top tier does not
    auto-merge because an extraordinary number is the suspicious one."""

    def test_the_gain_tiers(self):
        for delta, want in ((3.0, "eval:pass"), (9.9, "eval:pass"),
                            (10.1, "eval:major"), (24.9, "eval:major"),
                            (25.1, "eval:landmark"), (300.0, "eval:landmark")):
            self.assertEqual(bot.verdict(True, {16: delta})[0], want, delta)

    def test_the_regression_tiers(self):
        for delta, want in ((-1.9, "eval:noise"), (-2.1, "eval:slower"),
                            (-4.9, "eval:slower"), (-5.1, "eval:reject")):
            self.assertEqual(bot.verdict(True, {16: delta})[0], want, delta)

    def test_landmark_waits_for_a_human_but_the_gate_still_passes(self):
        action, why = bot.settle("eval:landmark", Settle.RECEIPT)
        self.assertEqual(action, "wait")
        self.assertIn("human", why)
        self.assertEqual(bot.STATUS_STATE["eval:landmark"], "success")

    def test_major_merges_only_when_the_rounds_agree(self):
        self.assertEqual(bot.settle("eval:major", Settle.RECEIPT, True)[0], "merge")
        action, why = bot.settle("eval:major", Settle.RECEIPT, False)
        self.assertEqual(action, "wait")
        self.assertIn("unstable", why)

    def test_slower_closes_the_pr_but_still_reads_as_a_trade(self):
        label, reason = bot.verdict(True, {16: -3.0})
        self.assertEqual(bot.STATUS_STATE[label], "failure")
        self.assertIn("fair trade", reason)
        self.assertEqual(bot.settle(label, Settle.RECEIPT)[0], "close")

    def test_every_label_has_a_colour_and_a_status(self):
        self.assertEqual(set(bot.EVAL_LABELS), set(bot.STATUS_STATE))


class DynamicBar(unittest.TestCase):
    """The bar is what this session earns — never looser than the rule."""

    def samples(self, *pr_vals):
        return {"pr": {"16": list(pr_vals)}, "main": {"16": [1000.0, 1000.0, 1000.0]}}

    def test_a_tight_session_keeps_the_published_floor(self):
        # real numbers: 1733.1 / 1733.5 / 1733.7 -> spread 0.035%
        bar = bot.scorer.effective_bar(self.samples(1733.1, 1733.5, 1733.7))
        self.assertEqual(bar, bot.NOISE_PCT)

    def test_a_jittery_session_raises_its_own_bar(self):
        bar = bot.scorer.effective_bar(self.samples(900.0, 1000.0, 1100.0))
        self.assertGreater(bar, bot.NOISE_PCT)
        self.assertAlmostEqual(bar, 60.0, places=1)

    def test_the_bar_never_drops_below_the_floor(self):
        self.assertGreaterEqual(bot.scorer.effective_bar({}), bot.NOISE_PCT)
        self.assertGreaterEqual(
            bot.scorer.effective_bar(self.samples(1000.0)), bot.NOISE_PCT)

    def test_a_raised_bar_downgrades_a_marginal_claim(self):
        b = synthetic_bundle(samples={
            "pr": {"16": [1030.0, 1100.0, 970.0], "64": [5000.0, 5000.0, 5000.0]},
            "main": {"16": [1000.0, 1000.0, 1000.0], "64": [5000.0, 5000.0, 5000.0]}})
        v = bot.scorer.score_bundle(b)
        self.assertGreater(v["bar_pct"], 3.0)
        self.assertEqual(v["label"], "eval:noise")


class Shape(unittest.TestCase):
    def test_shapes(self):
        bar = 2.0
        self.assertEqual(bot.scorer.shape({"16": 8.0, "64": 8.4}, bar), "uniform")
        self.assertEqual(bot.scorer.shape({"16": 0.5, "64": 16.0}, bar), "batch-skewed")
        self.assertEqual(bot.scorer.shape({"16": 16.0, "64": 0.5}, bar), "latency-skewed")
        self.assertEqual(bot.scorer.shape({"16": 8.0, "64": -8.0}, bar), "mixed")
        self.assertEqual(bot.scorer.shape({"16": 0.1, "64": -0.2}, bar), "flat")
        self.assertEqual(bot.scorer.shape({}, bar), "n/a")


class Taint(unittest.TestCase):
    def test_engine_changes_do_not_taint(self):
        t, u = bot.classify_taint([("M", "braid/model/gdn.py"),
                                   ("A", "braid/kernels/csrc/scan.cu")])
        self.assertEqual((t, u), ([], []))

    def test_modified_bench_taints(self):
        t, _ = bot.classify_taint([("M", "braid/bench/decode_speed.py")])
        self.assertEqual(t, ["braid/bench/decode_speed.py"])

    def test_modified_or_deleted_test_taints(self):
        t, _ = bot.classify_taint([("M", "tests/test_gdn_decode_kernel.py")])
        self.assertTrue(t)
        t, _ = bot.classify_taint([("D", "tests/test_loader.py")])
        self.assertTrue(t)

    def test_weakened_reference_oracle_taints(self):
        t, _ = bot.classify_taint([("M", "braid/reference/gdn.py")])
        self.assertEqual(t, ["braid/reference/gdn.py"])

    def test_added_test_is_unexercised_not_tainted(self):
        t, u = bot.classify_taint([("A", "tests/test_new_kernel.py")])
        self.assertEqual(t, [])
        self.assertEqual(u, ["tests/test_new_kernel.py"])

    def test_touching_the_bot_or_workflows_taints(self):
        t, _ = bot.classify_taint([("M", "scripts/pr_eval_bot.py")])
        self.assertTrue(t)
        t, _ = bot.classify_taint([("A", ".github/workflows/evil.yml")])
        self.assertTrue(t)

    def test_prefix_is_a_directory_not_a_substring(self):
        t, u = bot.classify_taint([("M", "braid/benchmarks_notes.md"),
                                   ("M", "tests_helper.py")])
        self.assertEqual((t, u), ([], []))

    def test_other_scripts_do_not_taint(self):
        t, _ = bot.classify_taint([("M", "scripts/remote.sh")])
        self.assertEqual(t, [])


class OverlayHarness(unittest.TestCase):
    @staticmethod
    def _write(root, rel, text):
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(text)

    def test_pinned_dirs_come_from_base_wholesale(self):
        with tempfile.TemporaryDirectory() as tmp:
            pr, base = os.path.join(tmp, "pr"), os.path.join(tmp, "base")
            self._write(pr, "braid/model/gdn.py", "PR ENGINE")
            self._write(pr, "braid/bench/decode_speed.py", "CHEAT")
            self._write(pr, "tests/test_x.py", "WEAKENED")
            self._write(pr, "tests/conftest.py", "MONKEYPATCH")
            self._write(base, "braid/bench/decode_speed.py", "TRUSTED")
            self._write(base, "tests/test_x.py", "TRUSTED")
            bot.overlay_harness(pr, base)
            read = lambda rel: open(os.path.join(pr, rel)).read()  # noqa: E731
            self.assertEqual(read("braid/model/gdn.py"), "PR ENGINE")
            self.assertEqual(read("braid/bench/decode_speed.py"), "TRUSTED")
            self.assertEqual(read("tests/test_x.py"), "TRUSTED")
            # wholesale replace: a PR-added conftest must NOT survive into
            # the pinned suite's process
            self.assertFalse(os.path.exists(os.path.join(pr, "tests/conftest.py")))

    def test_dir_missing_in_base_is_removed_from_pr(self):
        with tempfile.TemporaryDirectory() as tmp:
            pr, base = os.path.join(tmp, "pr"), os.path.join(tmp, "base")
            self._write(pr, "braid/reference/new_oracle.py", "PR ORACLE")
            os.makedirs(base, exist_ok=True)
            bot.overlay_harness(pr, base)
            self.assertFalse(os.path.exists(os.path.join(pr, "braid/reference")))


class TaintedVerdict(unittest.TestCase):
    def test_taint_caps_a_pass_at_tainted(self):
        label, reason = bot.verdict(True, {16: 0.4, 64: 5.2},
                                    tainted=["braid/bench/decode_speed.py"])
        self.assertEqual(label, "eval:tainted")
        self.assertIn("harness", reason)

    def test_taint_blocks_a_flat_result_too(self):
        # the red-team case: rig the bench, measure noise. A green,
        # mergeable status here would defeat the whole taint policy.
        label, reason = bot.verdict(True, {16: 0.2},
                                    tainted=["braid/bench/decode_speed.py"])
        self.assertEqual(label, "eval:tainted")
        self.assertIn("decode_speed.py", reason)
        self.assertEqual(bot.STATUS_STATE[label], "failure")

    def test_reject_still_outranks_taint(self):
        self.assertEqual(bot.verdict(True, {16: -9.0}, tainted=["tests/t.py"])[0],
                         "eval:reject")
        self.assertEqual(bot.verdict(False, {}, tainted=["tests/t.py"])[0],
                         "eval:reject")

    def test_untainted_pass_is_still_a_pass(self):
        self.assertEqual(bot.verdict(True, {16: 5.0}, tainted=[])[0], "eval:pass")


def synthetic_bundle(**kw):
    base = {
        "schema": bot.scorer.SCHEMA_IN,
        "pr": 7, "head": "a" * 40, "eval_sha": "b" * 40, "base_sha": "c" * 40,
        "mode": "merge-vs-main", "arm": "graphed-kvbucket", "batches": [16, 64],
        "reps": 3, "noise_pct": 2.0, "box": "47055458", "tests_ok": True,
        "suite_tail_sha256": "0" * 64,
        "samples": {"pr": {"16": [820.0, 825.0, 818.0], "64": [5400.0, 5390.0, 5410.0]},
                    "main": {"16": [800.0, 802.0, 799.0], "64": [5100.0, 5105.0, 5095.0]}},
        "name_status": [["M", "braid/model/gdn.py"]],
        "integrity": {"model_manifest": "m" * 64, "main_ext_hash": "e" * 64},
    }
    base.update(kw)
    return base


class Scorer(unittest.TestCase):
    def test_policy_is_single_sourced(self):
        # the bot must not grow its own copies — same function objects
        self.assertIs(bot.verdict, bot.scorer.verdict)
        self.assertIs(bot.classify_taint, bot.scorer.classify_taint)
        self.assertEqual(bot.NOISE_PCT, bot.scorer.NOISE_PCT)

    def test_score_bundle_pass(self):
        v = bot.scorer.score_bundle(synthetic_bundle())
        self.assertEqual(v["label"], "eval:pass")
        self.assertEqual(v["deltas_pct"].keys(), {"16", "64"})
        self.assertAlmostEqual(v["deltas_pct"]["64"], 5.882, places=3)

    def test_score_bundle_taint_caps_pass(self):
        b = synthetic_bundle(name_status=[["M", "braid/bench/decode_speed.py"]])
        self.assertEqual(bot.scorer.score_bundle(b)["label"], "eval:tainted")

    def test_score_bundle_suite_failure(self):
        v = bot.scorer.score_bundle(synthetic_bundle(tests_ok=False, samples={}))
        self.assertEqual(v["label"], "eval:reject")
        self.assertEqual(v["medians"], {})

    def test_scorer_cli_matches_in_process(self):
        # what the TEE prints must equal what the bot computes locally
        import subprocess
        import sys
        bundle = synthetic_bundle()
        expected = bot.scorer.canonical(bot.scorer.score_bundle(bundle)) + "\n"
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "bundle.json")
            with open(p, "w") as f:
                json.dump(bundle, f)
            out = subprocess.run([sys.executable, "-B", str(bot._SCORER_PATH), p],
                                 capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout, expected)

    def test_canonical_is_stable_under_key_order(self):
        a = bot.scorer.canonical({"b": 1, "a": [1, 2]})
        b = bot.scorer.canonical({"a": [1, 2], "b": 1})
        self.assertEqual(a, b)
        self.assertNotIn(" ", a)

    def test_build_bundle_round_trips_through_scorer(self):
        samples = {"pr": {16: [820.0], 64: [5400.0]}, "main": {16: [800.0], 64: [5100.0]}}
        bundle = bot.build_bundle(9, "d" * 40, "e" * 40, "f" * 40, "merge-vs-main",
                                  [16, 64], 1, True, samples,
                                  [("A", "tests/test_new.py")], "suite tail text",
                                  "m" * 64, None)
        v = bot.scorer.score_bundle(bundle)
        self.assertEqual(v["label"], "eval:pass")
        self.assertEqual(v["unexercised"], ["tests/test_new.py"])
        # canonicalizable (json round-trip identical)
        s = bot.scorer.canonical(bundle)
        self.assertEqual(bot.scorer.canonical(json.loads(s)), s)

    def test_touching_the_scorer_itself_taints(self):
        t, _ = bot.classify_taint([("M", "scripts/eval_scorer.py")])
        self.assertTrue(t)


class InfraFailureClassification(unittest.TestCase):
    """A gate must not blame a contributor for the box running out of VRAM."""

    def test_torch_oom_is_infrastructure(self):
        self.assertTrue(bot.scorer.suite_infra_failure(
            "ERROR tests/test_full_forward.py::test_capital_of_france - "
            "torch.OutOfMemoryError: CUDA out of memory"))

    def test_conftest_vram_budget_is_infrastructure(self):
        self.assertTrue(bot.scorer.suite_infra_failure(
            "RuntimeError: no 4-byte stack with both mixer types fits in "
            "2.7 GiB for 32 layers"))

    def test_disk_exhaustion_is_infrastructure(self):
        self.assertTrue(bot.scorer.suite_infra_failure(
            "OSError: [Errno 28] No space left on device"))

    def test_a_real_assertion_failure_is_not_infrastructure(self):
        self.assertFalse(bot.scorer.suite_infra_failure(
            "FAILED tests/test_gdn_decode_kernel.py::test_slot_parity - "
            "assert Tensor(1.2e-3) < 1e-5\n1 failed, 310 passed"))

    def test_a_clean_run_is_not_infrastructure(self):
        self.assertFalse(bot.scorer.suite_infra_failure("311 passed in 339.26s"))


class DurableEvidence(unittest.TestCase):
    def test_every_label_maps_to_a_commit_status_state(self):
        self.assertEqual(set(bot.STATUS_STATE), set(bot.EVAL_LABELS))
        self.assertTrue(set(bot.STATUS_STATE.values())
                        <= {"success", "failure", "error", "pending"})

    def test_tainted_and_reject_block_a_required_status(self):
        self.assertEqual(bot.STATUS_STATE["eval:tainted"], "failure")
        self.assertEqual(bot.STATUS_STATE["eval:reject"], "failure")
        self.assertEqual(bot.STATUS_STATE["eval:pass"], "success")


class OwnerOnlyPaths(unittest.TestCase):
    """CODEOWNERS states the owner-only list; owner-only.yml enforces it.
    Two copies of a security-relevant list drift, so this pins them together
    — same idiom as the Makefile and PR-template consistency tests below."""

    ROOT = pathlib.Path(__file__).resolve().parent.parent

    def owners(self) -> set[str]:
        text = (self.ROOT / ".github" / "CODEOWNERS").read_text()
        out = set()
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("/"):
                out.add(line.split()[0].lstrip("/"))
        return out

    def workflow(self) -> tuple[set[str], set[str]]:
        import re
        text = (self.ROOT / ".github" / "workflows" / "owner-only.yml").read_text()
        def arr(name):
            m = re.search(name + r"\s*=\s*\[(.*?)\]", text, re.S)
            self.assertIsNotNone(m, f"{name} not found in owner-only.yml")
            return set(re.findall(r"'([^']+)'", m.group(1)))
        return arr("FILES"), arr("PREFIXES")

    def test_codeowners_and_the_workflow_describe_the_same_set(self):
        files, prefixes = self.workflow()
        self.assertEqual(self.owners(), files | prefixes)

    def test_the_eval_infrastructure_is_owner_only(self):
        """The grader must never be editable by a PR it will grade."""
        files, prefixes = self.workflow()
        for path in bot.scorer.INFRA_FILES:
            self.assertIn(path, files, path)
        for pre in bot.scorer.INFRA_PREFIXES:
            self.assertIn(pre, prefixes, pre)

    def test_the_bench_harness_is_owner_only(self):
        self.assertIn("braid/bench/", self.workflow()[1])

    def test_prefixes_end_in_a_slash(self):
        """`braid/bench` without the slash would also match
        `braid/benchmarks.py`. The bot's own taint check learned this."""
        for pre in self.workflow()[1]:
            self.assertTrue(pre.endswith("/"), pre)

    def test_pinned_oracles_are_deliberately_not_owner_only(self):
        """braid/reference/ and tests/ are pinned during the eval and already
        cap a verdict at eval:tainted, which the outcome policy closes. Adding
        them here would be redundant; this test makes that a decision someone
        has to change on purpose rather than drift into."""
        files, prefixes = self.workflow()
        for path in ("braid/reference/", "tests/"):
            self.assertNotIn(path, prefixes, path)
            self.assertNotIn(path.rstrip("/"), files, path)

    def test_the_workflow_never_runs_pr_code(self):
        text = (self.ROOT / ".github" / "workflows" / "owner-only.yml").read_text()
        self.assertIn("pull_request_target", text)
        self.assertNotIn("actions/checkout", text,
                         "a pull_request_target job with a write token must "
                         "never check out PR code")


class BoxCapacity(unittest.TestCase):
    """vast.ai refuses a start with exit 0 and prose. Reading that is the
    difference between 'no GPU free right now' and seven minutes of silence
    ending in a misleading 'ssh never came up'."""

    # verbatim, instance 47055458, 2026-08-11T20:27:55+09:00
    REAL = "Required resources are currently unavailable, state change queued.\n"

    def test_the_real_refusal_is_recognised(self):
        self.assertIsNotNone(bot.start_was_refused(self.REAL))

    def test_the_reason_is_carried_not_swallowed(self):
        why = bot.start_was_refused(self.REAL)
        self.assertIn("currently unavailable", why)

    def test_a_normal_start_is_not_a_refusal(self):
        for ok in ("starting instance 47055458", "success", "", None):
            self.assertIsNone(bot.start_was_refused(ok), repr(ok))

    def test_matching_is_case_insensitive(self):
        self.assertIsNotNone(bot.start_was_refused(self.REAL.upper()))

    def test_unavailable_is_not_a_generic_failure(self):
        """main() must be able to tell capacity apart from a real fault, so
        a busy host never reaches a contributor as eval:error."""
        self.assertTrue(issubclass(bot.BoxUnavailable, RuntimeError))


class SshEndpoints(unittest.TestCase):
    """Regression: the 19:30 poll on 2026-08-11 stopped here. `vastai ssh-url`
    reported the direct route, the old code REPLACED the working proxy with
    it, and the run spent seven minutes failing to reach a box that was up."""

    DEFAULT = ("root@ssh5.vast.ai", "15458")
    # exactly what instance 47055458 reported that evening
    INFO = {"ssh_host": "ssh5.vast.ai", "ssh_port": 15458,
            "public_ipaddr": "202.215.136.222", "direct_port_start": 52677}
    URL = "ssh://root@202.215.136.222:52677\n"

    def test_the_proxy_route_survives_a_direct_ssh_url(self):
        cands = bot.endpoint_candidates(self.URL, self.INFO, self.DEFAULT)
        self.assertIn(("root@ssh5.vast.ai", "15458"), cands)
        self.assertIn(("root@202.215.136.222", "52677"), cands)

    def test_ssh_url_is_tried_first_but_is_not_the_only_one(self):
        cands = bot.endpoint_candidates(self.URL, self.INFO, self.DEFAULT)
        self.assertEqual(cands[0], ("root@202.215.136.222", "52677"))
        self.assertGreater(len(cands), 1)

    def test_the_default_is_always_present_and_never_first(self):
        cands = bot.endpoint_candidates("", {}, self.DEFAULT)
        self.assertEqual(cands, [self.DEFAULT])
        cands = bot.endpoint_candidates(self.URL, self.INFO, self.DEFAULT)
        self.assertIn(self.DEFAULT, cands)
        self.assertNotEqual(cands[0], self.DEFAULT)

    def test_no_duplicates(self):
        cands = bot.endpoint_candidates(self.URL, self.INFO, self.DEFAULT)
        self.assertEqual(len(cands), len(set(cands)))

    def test_a_bare_host_gets_the_root_user(self):
        cands = bot.endpoint_candidates("", {"ssh_host": "ssh9.vast.ai",
                                             "ssh_port": 40}, self.DEFAULT)
        self.assertIn(("root@ssh9.vast.ai", "40"), cands)

    def test_garbage_never_crashes_the_boot(self):
        for url in ("", "not a url", "ssh://nohost", None):
            self.assertTrue(bot.endpoint_candidates(url, {}, self.DEFAULT))


class Settle(unittest.TestCase):
    """A verdict ends the PR: merged, or closed with the reason. The only
    things that wait are the ones that are OUR fault, not the contributor's."""

    RECEIPT = {"verification": {"intel_verified": True}}

    def test_attested_pass_merges(self):
        action, why = bot.settle("eval:pass", self.RECEIPT)
        self.assertEqual(action, "merge", why)

    def test_attested_noise_merges(self):
        # measured harmless is a merge condition too: requiring a speedup
        # would mean a bug fix or refactor could never merge itself.
        action, why = bot.settle("eval:noise", self.RECEIPT)
        self.assertEqual(action, "merge", why)

    def test_measured_rejections_close_with_a_reason(self):
        for label in ("eval:reject", "eval:slower", "eval:tainted"):
            action, why = bot.settle(label, self.RECEIPT)
            self.assertEqual(action, "close", label)
            self.assertTrue(why and not why.endswith("."), label)

    def test_every_closing_label_is_a_real_label_and_reads_as_failure(self):
        for label in bot.CLOSING_LABELS:
            self.assertIn(label, bot.EVAL_LABELS)
            self.assertEqual(bot.STATUS_STATE[label], "failure", label)

    def test_merge_and_close_sets_do_not_overlap(self):
        self.assertFalse(set(bot.MERGEABLE_LABELS) & set(bot.CLOSING_LABELS))

    def test_every_verdict_has_an_outcome(self):
        """No label may fall off the end of the policy unclassified."""
        for label in bot.EVAL_LABELS:
            action, why = bot.settle(label, self.RECEIPT)
            self.assertIn(action, ("merge", "close", "wait"), label)
            self.assertTrue(why, label)

    def test_an_eval_error_never_closes_the_pr(self):
        # The eval broke, not the PR. PR #5 is the standing example: it was
        # labelled eval:reject twice for 'pinned test suite failed' and the
        # verdict was withdrawn — the box was at fault. Closing on our own
        # infrastructure would have killed a good PR twice.
        action, why = bot.settle("eval:error", self.RECEIPT)
        self.assertEqual(action, "wait")
        self.assertIn("evaluation", why)

    def test_landmark_waits_for_a_human_but_the_gate_still_passes(self):
        action, why = bot.settle("eval:landmark", self.RECEIPT)
        self.assertEqual(action, "wait")
        self.assertIn("human", why)
        self.assertEqual(bot.STATUS_STATE["eval:landmark"], "success")

    def test_a_held_pr_is_never_closed(self):
        for label in bot.CLOSING_LABELS:
            action, _ = bot.settle(label, self.RECEIPT, None, ("hold",))
            self.assertEqual(action, "wait", label)

    def test_hold_also_blocks_a_merge(self):
        action, _ = bot.settle("eval:pass", self.RECEIPT, None, ("hold", "eval"))
        self.assertEqual(action, "wait")

    def test_a_missing_receipt_waits_it_does_not_close(self):
        # An attestation outage is our problem; blaming the contributor for
        # it by closing their PR is the failure mode this guards.
        for receipt in (None, {}, {"verification": {"intel_verified": False}}):
            for label in bot.MERGEABLE_LABELS:
                action, _ = bot.settle(label, receipt)
                self.assertEqual(action, "wait", (label, receipt))

    def test_major_merges_only_when_the_rounds_agree(self):
        self.assertEqual(bot.settle("eval:major", self.RECEIPT, True)[0], "merge")
        action, why = bot.settle("eval:major", self.RECEIPT, False)
        self.assertEqual(action, "wait")
        self.assertIn("unstable", why)

    def test_an_unstable_round_waits_rather_than_closing(self):
        # Our two rounds disagreeing is not evidence against the PR.
        self.assertEqual(bot.settle("eval:pass", self.RECEIPT, False)[0], "wait")

    def test_the_close_comment_names_the_verdict_and_the_way_back(self):
        body = bot.close_comment("eval:reject", bot.CLOSING_LABELS["eval:reject"])
        self.assertIn("eval:reject", body)
        self.assertIn("reopen", body)
        self.assertIn("hold", body)
        self.assertIn(bot.CLOSING_LABELS["eval:reject"], body)


class HarnessCrossCheck(unittest.TestCase):
    """A bench change can clear itself by measurement; a test change cannot."""

    def cc(self, **rows):
        return {"ran": True, "reps": 1, "added": [],
                "shared": {k: {"pinned": v[0], "pr_harness": v[1],
                               "delta_pct": (v[1] - v[0]) / v[0] * 100}
                           for k, v in rows.items()}}

    def test_only_bench_paths_are_crosscheckable(self):
        self.assertTrue(bot.scorer.crosscheckable(["braid/bench/decode_speed.py"]))
        self.assertFalse(bot.scorer.crosscheckable(["tests/test_x.py"]))
        self.assertFalse(bot.scorer.crosscheckable(["braid/reference/gdn.py"]))
        self.assertFalse(bot.scorer.crosscheckable(
            ["braid/bench/decode_speed.py", "tests/test_x.py"]))
        self.assertFalse(bot.scorer.crosscheckable([]))

    def test_agreeing_harness_clears(self):
        ok, why = bot.scorer.crosscheck_agrees(
            self.cc(**{"graphed-kvbucket@16": (1733.0, 1734.0)}))
        self.assertTrue(ok, why)

    def test_the_red_team_inflation_is_caught(self):
        ok, why = bot.scorer.crosscheck_agrees(
            self.cc(**{"graphed-kvbucket@16": (1733.0, 1733.0 * 1.5)}))
        self.assertFalse(ok)
        self.assertIn("50.0%", why)

    def test_no_shared_configuration_does_not_clear(self):
        self.assertFalse(bot.scorer.crosscheck_agrees(
            {"ran": True, "shared": {}, "added": ["x@128"]})[0])
        self.assertFalse(bot.scorer.crosscheck_agrees(None)[0])

    def test_cleared_bench_pr_can_reach_pass(self):
        b = synthetic_bundle(name_status=[["M", "braid/bench/decode_speed.py"]],
                             crosscheck=self.cc(**{"graphed-kvbucket@16": (1733.0, 1733.5)}))
        v = bot.scorer.score_bundle(b)
        self.assertEqual(v["label"], "eval:pass")
        self.assertTrue(v["crosscheck_ok"])
        self.assertEqual(bot.settle(v["label"], Settle.RECEIPT)[0], "merge")

    def test_disagreeing_bench_pr_stays_tainted(self):
        b = synthetic_bundle(name_status=[["M", "braid/bench/decode_speed.py"]],
                             crosscheck=self.cc(**{"graphed-kvbucket@16": (1733.0, 2599.5)}))
        v = bot.scorer.score_bundle(b)
        self.assertEqual(v["label"], "eval:tainted")
        # A bench change whose own numbers disagree with the pinned harness
        # is exactly PR #5: it closes, it does not quietly merge.
        self.assertEqual(bot.settle(v["label"], Settle.RECEIPT)[0], "close")

    def test_a_test_change_is_never_cleared(self):
        b = synthetic_bundle(name_status=[["M", "tests/test_x.py"]],
                             crosscheck=self.cc(**{"graphed-kvbucket@16": (1733.0, 1733.0)}))
        v = bot.scorer.score_bundle(b)
        self.assertEqual(v["label"], "eval:tainted")
        self.assertFalse(v["crosscheck_ok"])


class DocsOnlyAutoMerge(unittest.TestCase):
    def info(self, **kw):
        base = {"state": "OPEN", "isDraft": False, "labels": [],
                "files": [{"path": "docs/ROADMAP.md"}]}
        base.update(kw)
        return base

    def files(self, *paths):
        return {"files": [{"path": p} for p in paths]}

    def test_the_bot_itself_never_docs_merges(self):
        # 'not a runtime path' is true of the grader, the workflows, the
        # Makefile and pyproject — none of which may merge unread.
        for p in ("scripts/pr_eval_bot.py", "scripts/eval_scorer.py",
                  ".github/workflows/gate.yml", ".github/PULL_REQUEST_TEMPLATE.md",
                  "Makefile", "pyproject.toml", ".gitignore"):
            ok, why = bot.docs_only_automerge(self.info(**self.files(p)), self.GREEN)
            self.assertFalse(ok, f"{p} must not auto-merge")
            self.assertIn("non-prose", why)

    def test_prose_paths_qualify(self):
        for p in ("docs/ROADMAP.md", "README.md", "CONTRIBUTING.md", "LICENSE"):
            self.assertTrue(bot.inert_paths([p]), p)

    def test_one_bad_path_disqualifies_the_whole_pr(self):
        ok, _ = bot.docs_only_automerge(
            self.info(**self.files("README.md", "scripts/pr_eval_bot.py")), self.GREEN)
        self.assertFalse(ok)

    def test_a_non_md_file_under_docs_does_not_qualify(self):
        self.assertFalse(bot.inert_paths(["docs/bench.py"]))
        self.assertFalse(bot.inert_paths([]))

    GREEN = [{"name": "ruff", "state": "SUCCESS"}, {"name": "gate", "state": "SUCCESS"}]

    def test_all_green_merges(self):
        ok, why = bot.docs_only_automerge(self.info(), self.GREEN)
        self.assertTrue(ok, why)

    def test_a_failing_check_blocks(self):
        ok, _ = bot.docs_only_automerge(
            self.info(), self.GREEN + [{"name": "review", "state": "FAILURE"}])
        self.assertFalse(ok)

    def test_a_pending_check_waits(self):
        ok, why = bot.docs_only_automerge(
            self.info(), self.GREEN + [{"name": "review", "state": "PENDING"}])
        self.assertFalse(ok)
        self.assertIn("still running", why)

    def test_our_own_not_required_stamp_is_ignored(self):
        ok, _ = bot.docs_only_automerge(
            self.info(), self.GREEN + [{"name": bot.STATUS_CONTEXT, "state": "SUCCESS"}])
        self.assertTrue(ok)

    def test_draft_or_hold_or_no_checks_block(self):
        self.assertFalse(bot.docs_only_automerge(self.info(isDraft=True), self.GREEN)[0])
        self.assertFalse(bot.docs_only_automerge(
            self.info(labels=[{"name": "hold"}]), self.GREEN)[0])
        self.assertFalse(bot.docs_only_automerge(self.info(), [])[0])


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


class ReproducibleBenchConfig(unittest.TestCase):
    """A contributor must be able to measure what the bot measures.

    The eval reads one arm at two batches with a specific prompt length,
    quant set and state dtype. If `make bench-eval` and the PR template drift
    from that, a contributor benches a nearby configuration, sees a different
    delta, and has no way to explain the disagreement with the posted verdict.
    So the Makefile is required to ASK the bot for the command rather than
    keep a copy, and these tests fail the moment a copy reappears.
    """
    ROOT = pathlib.Path(__file__).resolve().parent.parent

    def test_human_and_json_forms_differ_only_by_json(self):
        human = bot.bench_cmd(bot.EVAL_BATCHES, json_out=False)
        self.assertEqual(bot.bench_cmd(bot.EVAL_BATCHES), human + " --json")

    def test_printed_command_names_the_scored_batches(self):
        human = bot.bench_cmd(bot.EVAL_BATCHES, json_out=False)
        self.assertIn("--batches 16 64", human)
        self.assertIn("braid.bench.decode_speed", human)

    def test_makefile_derives_the_command_instead_of_copying_it(self):
        mk = (self.ROOT / "Makefile").read_text()
        self.assertIn("bench-eval:", mk)
        self.assertIn("--print-bench-cmd", mk)
        recipe = mk.split("bench-eval:", 1)[1].split("\n\n", 1)[0]
        self.assertNotIn("decode_speed", recipe,
                         "bench-eval must ask the bot for the command, not repeat it")

    def test_pr_template_points_at_bench_eval_and_names_the_arm(self):
        tpl = (self.ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text()
        self.assertIn("make bench-eval", tpl)
        self.assertIn(bot.ARM, tpl)
        self.assertNotIn("make bench-scaling", tpl,
                         "scan_scaling is a different benchmark; it is not what is scored")

    def test_contributing_names_the_arm_and_the_batches(self):
        doc = (self.ROOT / "CONTRIBUTING.md").read_text()
        self.assertIn("make bench-eval", doc)
        self.assertIn(bot.ARM, doc)
        for b in bot.EVAL_BATCHES:
            self.assertIn(f"B={b}", doc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
