#!/usr/bin/env python3
"""Verify a braid eval receipt from polaris.computer — offline, no secrets.

Anyone with a clean clone can check that a published verdict was produced
by the committed scorer policy from the recorded evidence bundle inside a
genuine Intel TDX machine. The receipt (JSON, stored verbatim in the PR's
git note under refs/notes/braid-eval) carries a raw DCAP quote whose
64-byte report_data binds everything; this tool recomputes each binding
per the recipe the receipt itself declares:

  report_data[0:32]  = sha256(nonce_hex || e2e_pubkey_b64)
  report_data[32:64] = sha256(bound_digest || result_sha256 || egress_log_sha
                              || files_sha || policy_sha || artifacts_sha)
  bound_digest       = "sha256:" + sha256(workload script)   (workload mode)
  files_sha          = sha256(concat of sorted "target\\tsha256\\n" lines)
  result_sha256      = sha256(stdout)

report_data lives at byte offset 568 in the TDX v4 quote (the tail of the
584-byte TD report body) — verified empirically against live receipts.

Checks performed:
  1. stdout hash matches result_sha256 (the verdict cannot have been edited)
  2. bound_digest matches the expected workload command
  3. files_sha256 matches the bundle + the scorer file in THIS clone
     (i.e. the policy that ran is the policy committed here)
  4. both report_data halves recompute and appear at offset 568 in the quote
  5. the scorer, re-run locally on the bundle, reproduces the receipt's
     stdout byte-for-byte (verdict is a pure function of the bundle)

NOT yet checked here: the DCAP signature chain from the quote to Intel's
root CA (the `verification.collateral_urls` in the receipt list the Intel
PCS endpoints; polaris.computer's `intel_verified` asserts it, and the
collateral is embedded for independent checking with standard DCAP tooling
such as intel/SGXDataCenterAttestationPrimitives' QVL). Wiring a local QVL
pass is the remaining gap between "bindings verified" and "silicon
verified" — tracked in docs/TDX_ATTESTATION_SPEC.md.

Usage:
  python3 scripts/verify_receipt.py receipt.json bundle.json
  python3 scripts/verify_receipt.py receipt.json bundle.json --scorer scripts/eval_scorer.py
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import pathlib
import subprocess
import sys

REPORT_DATA_OFFSET = 568
WORKLOAD = "python3 /in/eval_scorer.py /in/bundle.json"
SCORER_TARGET = "/in/eval_scorer.py"
BUNDLE_TARGET = "/in/bundle.json"


def sha_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("receipt", help="receipt JSON as returned by /v1/attest")
    ap.add_argument("bundle", help="the canonical evidence bundle JSON")
    ap.add_argument("--scorer", default=str(pathlib.Path(__file__).parent / "eval_scorer.py"),
                    help="scorer file to verify against (default: this clone's)")
    args = ap.parse_args()

    receipt = json.loads(pathlib.Path(args.receipt).read_bytes())
    bundle_bytes = pathlib.Path(args.bundle).read_bytes()
    scorer_bytes = pathlib.Path(args.scorer).read_bytes()
    t = receipt["tee_attestation"]
    quote = base64.b64decode(t["quote_b64"])
    stdout = base64.b64decode(t["stdout_b64"]) if t.get("stdout_b64") \
        else receipt.get("stdout", "").encode()

    ok = True

    # 1. verdict integrity
    ok &= check("result_sha256 == sha256(stdout)",
                t["result_sha256"] == sha_hex(stdout))

    # 2. what ran
    ok &= check("bound_digest == sha256(expected workload)",
                t["bound_digest"] == "sha256:" + sha_hex(WORKLOAD.encode()),
                WORKLOAD)

    # 3. policy + evidence identity: the files the TEE saw are the bundle
    #    on record and the scorer committed in this clone
    lines = sorted([f"{BUNDLE_TARGET}\t{sha_hex(bundle_bytes)}\n",
                    f"{SCORER_TARGET}\t{sha_hex(scorer_bytes)}\n"])
    files_sha = sha_hex("".join(lines).encode())
    ok &= check("files_sha256 == sha256(bundle + this clone's scorer)",
                receipt.get("files_sha256") == files_sha)

    # 4. hardware binding: both report_data halves recompute and sit at the
    #    fixed offset inside the raw TD quote
    h1 = sha_hex((t["nonce"] + t["e2e_pubkey_b64"]).encode())
    h2 = sha_hex((t["bound_digest"] + t["result_sha256"]
                  + receipt.get("egress_log_sha256", "")
                  + receipt.get("files_sha256", "")
                  + receipt.get("policy_sha256", "")
                  + receipt.get("artifacts_sha256", "")).encode())
    rd = quote[REPORT_DATA_OFFSET:REPORT_DATA_OFFSET + 64]
    ok &= check("report_data[0:32] == sha256(nonce || pubkey)",
                rd[:32] == bytes.fromhex(h1))
    ok &= check("report_data[32:64] == sha256(digest||result||egress||files||…)",
                rd[32:] == bytes.fromhex(h2))

    # 5. determinism: the committed scorer reproduces the attested verdict
    local = subprocess.run([sys.executable, "-B", args.scorer, args.bundle],
                           capture_output=True, timeout=60)
    ok &= check("local scorer reproduces attested stdout byte-for-byte",
                local.returncode == 0 and local.stdout == stdout)

    print()
    if ok:
        print("OK — verdict, policy, evidence and hardware binding all verify.")
        print("(DCAP chain-to-Intel-root not checked by this tool; collateral "
              "is embedded in the receipt for standard QVL tooling.)")
        return 0
    print("VERIFICATION FAILED — do not trust this receipt.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
