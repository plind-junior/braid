## Summary

<!-- What this changes and why. One or two lines. -->


## If this PR claims a speedup

> Numbers are the part of this project most easily damaged by carelessness.
> Every number below must be **measured** — a real run on an RTX 5090 with the
> method stated — or clearly labelled **projected**. A delta that does not
> clear the noise floor (`make bench-noise`), or that compares against a
> different session's baseline, is not a result.
>
> Fill the table from **`make bench-eval`** — the eval bot's exact
> configuration. Read the **`graphed-kvbucket`** rows: that is the arm the
> verdict is computed from, and a number from any other arm will not agree
> with the one the bot posts on this PR.

- [ ] Tested on **RTX 5090** — `make test-remote` passes on this branch

| `graphed-kvbucket` | decode tok/s (B=16) | decode tok/s (B=64) |
|---|--:|--:|
| before (main) | | |
| after (this PR) | | |

```text
# paste the bench output backing the table (make bench-eval), before -> after
```

## If this PR touches a kernel (`braid/kernels/`)

- [ ] Parity test against `braid/reference/` added or extended
- [ ] No `--use_fast_math` (it breaks fp32 oracle parity)
- [ ] Decode path stays capture-safe (no allocations or host syncs in the captured region)

<!-- Docs-only or non-perf changes: delete the sections above. Note the CI
     gate: a PR touching braid/ or tests/ must tick the RTX 5090 box — CI has
     no GPU, so the ticked box is the only signal the suite was run at all. -->
