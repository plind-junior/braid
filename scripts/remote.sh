#!/usr/bin/env bash
set -euo pipefail
: "${BRAID_SSH_KEY:=$HOME/.ssh/rtx5090}"
: "${BRAID_SSH_HOST:=root@ssh5.vast.ai}"
: "${BRAID_SSH_PORT:=15458}"
: "${BRAID_REMOTE_DIR:=/root/braid}"

SSH_OPTS="-i $BRAID_SSH_KEY -p $BRAID_SSH_PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

rsync -az --delete \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
  -e "ssh $SSH_OPTS" \
  "$REPO_ROOT/" "$BRAID_SSH_HOST:$BRAID_REMOTE_DIR/"

# shellcheck disable=SC2029
ssh $SSH_OPTS "$BRAID_SSH_HOST" "cd $BRAID_REMOTE_DIR && $*"
