#!/usr/bin/env bash
# The eval bot's cron wrapper. System cron runs this every 30 minutes; the
# bot only spins the GPU when there is an un-evaluated eligible PR head, so a
# quiet repo costs nothing but the poll.
#
# flock: evals serialize on one box; a poll that finds the previous one still
# running must exit, not queue. Cron's PATH is minimal, so gh/vastai
# (~/.local/bin) are added explicitly. The bot always runs from the local
# main tree — PR code never executes on this machine.
#
# Install (idempotent):
#   ( crontab -l 2>/dev/null | grep -v pr_eval_cron.sh ; \
#     echo "*/30 * * * * $HOME/Dev/plind-junior/braid/scripts/pr_eval_cron.sh" ) | crontab -
set -uo pipefail
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin"
# Polaris API key for attested receipts (optional — evals run without it).
# The key lives outside the repo on purpose; never move it into the tree.
if [ -f "$HOME/.config/braid/polaris.env" ]; then
  . "$HOME/.config/braid/polaris.env"
  export BRAID_POLARIS_API_KEY
fi
cd "$(dirname "$0")/.."
LOG="${BRAID_EVAL_LOG:-$HOME/braid-pr-eval.log}"

exec 9>"$HOME/.braid-pr-eval.lock"
flock -n 9 || exit 0
{
  echo "=== $(date -Is) pr-eval poll"
  # -u: unbuffered, so a 30-minute eval's progress is readable in the log
  # while it runs, not only after it exits.
  python3 -B -u scripts/pr_eval_bot.py
  echo "=== exit $?"
} >>"$LOG" 2>&1
