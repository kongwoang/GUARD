#!/usr/bin/env bash
# Download the saved outputs of the frozen models into data/artifacts.
#
#   bash data/download.sh              # all seven benchmarks, about 2.4 GB
#   bash data/download.sh mosei ave    # only some of them
set -euo pipefail

URL="https://github.com/kongwoang/GUARD/releases/download/v1.0"
DEST="${GUARD_ARTIFACTS:-$(cd "$(dirname "$0")" && pwd)/artifacts}"
ALL="mosei iemocap opportunity ave ninapro ptbxl drugban"

[ $# -eq 0 ] && set -- $ALL
mkdir -p "$DEST"
for name in "$@"; do
  case " $ALL " in *" $name "*) ;; *) echo "unknown benchmark: $name" >&2; exit 1 ;; esac
  echo "downloading $name"
  curl -fL --retry 3 -o "$DEST/$name.zip" "$URL/$name.zip"
  unzip -qo "$DEST/$name.zip" -d "$DEST"
  rm "$DEST/$name.zip"
done
echo "saved to $DEST"
