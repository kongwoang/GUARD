#!/usr/bin/env bash
# Download the saved outputs of the frozen models into data/artifacts.
#
#   bash data/download.sh              # all seven benchmarks, about 2.5 GB
#   bash data/download.sh mosei ave    # only some of them
set -euo pipefail

REPO="${GUARD_ARTIFACT_REPO:-anonymous221206/GUARD_checkpoint}"
DEST="${GUARD_ARTIFACTS:-$(cd "$(dirname "$0")" && pwd)/artifacts}"

files() {
  case "$1" in
    mosei)       echo "mosei_cmad/*" ;;
    iemocap)     echo "iemocap_momke/*" ;;
    opportunity) echo "opportunity_dcl_v2/*" ;;
    ave)         echo "ave_av_att/dumps/*" ;;
    ninapro)     echo "ninapro_cnn/seed*/*" ;;
    ptbxl)       echo "ptbxl_resnet1d_wang/*" ;;
    drugban)     echo "drugban_processed/*" ;;
    *) echo "unknown benchmark: $1" >&2; exit 1 ;;
  esac
}

[ $# -eq 0 ] && set -- mosei iemocap opportunity ave ninapro ptbxl drugban
patterns=()
for name in "$@"; do
  patterns+=("$(files "$name")")
done

python - "$REPO" "$DEST" "${patterns[@]}" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo, dest, *patterns = sys.argv[1:]
snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest, allow_patterns=patterns)
print(f"saved to {dest}")
PY
