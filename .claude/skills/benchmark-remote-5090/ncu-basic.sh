#!/usr/bin/env bash
# Canonical ncu metric set for a braid kernel, run on the remote 5090.
#
#   ./.claude/skills/benchmark-remote-5090/ncu-basic.sh "regex:gdn_decode.*" \
#       python3 -B -m braid.bench.scan_scaling
#
# ncu serializes and replays each launch: these numbers are METRICS, not timing.
# Time with measure_graphed() or nsys instead.
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 <kernel-name-regex> <command...>" >&2
  exit 2
fi

KERNEL="$1"; shift
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

METRICS="\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,\
launch__shared_mem_per_block_static,\
launch__shared_mem_per_block_dynamic,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
smsp__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active"

# --launch-skip 3 discards the warmup launches that run at idle clocks.
exec "$REPO_ROOT/scripts/remote.sh" ncu \
  --kernel-name "$KERNEL" \
  --launch-skip 3 --launch-count 10 \
  --metrics "$METRICS" \
  --print-summary per-kernel \
  "$@"
