#!/usr/bin/env bash
# Run the suite one module per process.
#
# **Why this exists.** Several modules hold a module-scoped engine, and pytest
# keeps a module fixture alive until that module ends. At Qwen3.5-4B those are
# 8.4 GiB (bf16) and 16.8 GiB (fp32) and the allocator absorbs the churn; at
# Qwen3.5-9B they are 16.7 and 17.2 GiB, and the reserved-but-unallocated blocks
# left behind by one module OOM the next one's load. Every module passes on its
# own — verified — so the failure is resource hygiene across modules, not
# correctness.
#
# `tests/conftest.py` already reclaims at module boundaries and that is enough at
# 4B. Rather than restructure a dozen fixtures for the larger target, give each
# module its own process and let the OS do it. One process per module is also
# what the measurement contract asks for anyway.
#
#     bash scripts/test_isolated.sh                        # all modules
#     BRAID_MODEL_DIR=/root/models/Qwen3.5-9B bash scripts/test_isolated.sh
#     bash scripts/test_isolated.sh -k slot                # args passed through
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

pass=0; fail=0; failed=()
for f in tests/test_*.py; do
  out=$(python3 -B -m pytest "$f" -q --no-header "$@" 2>&1)
  line=$(printf '%s\n' "$out" | tail -1)
  if printf '%s\n' "$out" | grep -qE "^(FAILED|ERROR)|failed|error"; then
    fail=$((fail + 1)); failed+=("$f")
    printf '%-44s %s\n' "$f" "$line"
    printf '%s\n' "$out" | grep -E "^(FAILED|ERROR)" | head -6 | sed 's/^/    /'
  else
    pass=$((pass + 1))
    printf '%-44s %s\n' "$f" "$line"
  fi
done

echo
echo "modules: $pass passed, $fail failed"
if [ "$fail" -ne 0 ]; then
  printf 'failed modules: %s\n' "${failed[*]}"
  exit 1
fi
