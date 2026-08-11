# Attested PR evaluation — TDX receipt spec

Status: **proposed** (Phase B of the eval-trust roadmap). Phase A — pinned harness,
taint capping, arm isolation, durable evidence (statuses + `refs/notes/braid-eval`) —
shipped in `scripts/pr_eval_bot.py` on 2026-08-11.

## The one sentence that scopes everything

A TEE receipt can prove **the verdict was computed honestly from the submitted
measurements by the declared policy code** — it cannot prove **the RTX 5090 reported
honest measurements**, because the 5090 is a consumer part with no confidential-compute
mode (that exists only on H100/H200/Blackwell datacenter GPUs). SparkInfer's Polaris
receipt has exactly the same boundary. Renting TDX buys us parity with their ceiling,
not more.

## Threat model delta

| Trusted today (Phase A) | Trusted after Phase B |
|---|---|
| The maintainer's machine ran the real `pr_eval_bot.py` at the committed revision | Nothing on the maintainer's machine — the scorer's code identity is in the quote |
| The posted medians/deltas equal what the benches printed | Recomputed inside the TEE from the raw rep logs in the bundle |
| The verdict rule (±2%, regression-outranks, taint cap) was applied as documented | Enforced by the attested policy binary |
| The 5090 box ran unmodified kernels on honest silicon | **Still assumed** (see Non-goals) |

## Architecture

```
local bot (untrusted)                  TDX CVM (attested)              GitHub
─────────────────────                  ──────────────────              ──────
collect evidence bundle ──HTTPS(pinned)──▶ braid-eval-scorer
  suite tail, ALL rep stdouts,             recompute medians/deltas
  tree/cache/model hashes,                 rerun verdict policy
  shas, taint diff, box id                 quote: REPORTDATA =
                                             sha256(bundle ‖ verdict ‖ policy sha)
                       ◀── receipt ────    (TD quote + verdict JSON)
post comment/label/status ─────────────────────────────────────────▶ PR
append note: bundle + receipt ─────────────────────────────────────▶ refs/notes/braid-eval
```

1. **The scorer** is the pure-policy half of the bot (`pick_tok_s`, `median`,
   `verdict`, `classify_taint` — already side-effect-free and unit-tested) packaged as
   a ~50-line HTTP service in a container. The image is built reproducibly from a
   tagged revision of this repo; its digest is published in the README.
2. **The bundle** is canonical JSON (sorted keys, no floats-from-medians — raw rep
   values only). The scorer recomputes everything; the bot's own arithmetic becomes
   advisory display.
3. **The receipt** is the raw TD quote plus the scorer's verdict JSON. REPORTDATA
   (64 bytes, attacker-immutable, hardware-signed) binds `sha256(bundle_hash ||
   verdict_hash || policy_git_sha)`. MRTD/RTMRs bind the VM image → container digest.
4. **Publication**: the receipt rides in the same git note as the evidence record
   (append-only), and the PR comment links the note. `scripts/verify_receipt.py`
   (ships with the repo) checks: quote signature chain to Intel roots, measurement
   registers against the published image digest, REPORTDATA against the note's bundle,
   and that the verdict in the comment equals the verdict in the receipt.

## Infra decision — polaris.computer (chosen 2026-08-11)

Disambiguation: this is **polaris.computer** (rents TDX hardware directly, per-second
billing, self-serve card signup), NOT Fr0ntierX's "Polaris" (a software layer that
deploys onto your own hyperscaler account — useless here since hyperscaler signup was
the blocker). polaris.computer is almost certainly the "Polaris receipt" in the
SparkInfer story.

Its `POST /v1/attest` one-shot endpoint is our architecture, already built:

| Spec concept (above) | /v1/attest field |
|---|---|
| policy code identity (pinned scorer digest) | `image` → `bound_digest` (content digest) in report_data |
| evidence bundle hash | `files` (≤8 × ≤256 KiB, read-only) → `files_sha256` in report_data |
| verdict hash | scorer stdout → `result_sha256` in report_data |
| network isolation of the scorer | `egress: "none"`, logged and hash-bound |
| the TD quote | raw DCAP quote + collateral in the receipt |
| public verifiability | `GET /r/:id` + offline recompute of all bindings |

report_data layout (theirs): bytes 0–31 = sha256(nonce ‖ pubkey), bytes 32–63 =
sha256(bound_digest ‖ result_sha256 ‖ egress_log_sha ‖ files_sha). Identical binding
strength to our design; we adopt their layout instead of running our own CVM.

Consequences for the design above: no rented CVM, no HTTP scorer service — the scorer
becomes a **public container image** (policy code is open source anyway) invoked
per-eval via /v1/attest with the canonical bundle as an input file and `egress: none`.
The PR comment links the public receipt viewer (`/r/:id`); the git note stores the
receipt JSON verbatim so evidence survives even if polaris.computer disappears.
Constraint to respect: bundle ≤ 8 files × 256 KiB — full bench logs live in the git
note, the bundle carries the parsed rep values plus the logs' hashes.

Trust model note: polaris.computer operates the host, but the DCAP quote chains to
Intel and every binding is recomputable offline — the operator is outside the trust
base for verification, same as SparkInfer's ceiling. Cost: per-second CPU billing for
a seconds-long scorer run — pennies per eval, no idle spend; free tier (one 3-hour
sandbox) covers the entire bring-up before any card charge.

## Phase C (explicitly deferred): attested auto-merge

Receipt-gated merging — a GitHub App whose token lives *inside* the CVM merges when
`eval:pass` + receipt verifies + `enforce_admins=true` so the maintainer cannot merge
around it. Deferred because: braid is single-maintainer; the taint/manual-review path
must exist anyway; and the measurement itself is still unattested, so auto-merge would
launder the weakest link behind the strongest seal.

## Non-goals and residual risk (state them or they get forgotten)

- **The measurement is not attested.** The bundle's rep logs come from a root-owned
  vast.ai box running PR code. Phase A's integrity checks (checksum re-push, cache
  hash-lock, model manifest, GPU-idle assertion) raise the bar; a TOCTOU daemon
  remains possible. Honest mitigation if it ever matters: quorum — re-run the eval on
  a second independently rented 5090 and require agreement within noise.
- The receipt proves policy-at-revision; it does not prove the policy is *wise*. The
  ±2% bar and regression rule stay in reviewed source.

## Rollout gates

1. ✅ 2026-08-11 — scorer extracted to `scripts/eval_scorer.py` (bot imports it;
   `assertIs` tests enforce single-sourcing; no prior ledger existed to replay, so
   equivalence is enforced by construction: bot and TEE run the same file).
2. ✅ 2026-08-11 — end-to-end receipt on a synthetic bundle via `/v1/attest`
   (workload mode, not image mode: `bound_digest` = sha256 of the scorer invocation,
   `files_sha256` binds the scorer file + bundle). TEE stdout byte-identical to the
   local scorer. Cost $0.0009, 15.6 s on a warm box. Note: Cloudflare fronts the API
   and 403s the default urllib UA — the bot sends `braid-pr-eval-bot/1.0`.
3. ✅ 2026-08-11 — `scripts/verify_receipt.py` recomputes all bindings offline
   (result hash, bound digest, files hash, both report_data halves at quote offset
   568, scorer determinism) — 6/6 PASS. Remaining gap, tracked: DCAP signature chain
   to Intel's root is not yet checked locally (collateral is embedded in every
   receipt for standard QVL tooling); until then that one link rests on the
   receipt's `intel_verified` flag.
4. Open — first attested verdict on a REAL PR eval (Phase A smoke PRs), then mention
   receipts in CONTRIBUTING and the README. Also open: /v1/attest returns no public
   `/r/:id` viewer URL in workload mode — receipts are self-hosted in the git note,
   which is sufficient; a public viewer link is nice-to-have to chase with polaris.
