#!/usr/bin/env bash
# Downloads the wake word models (openWakeWord v0.5.1, "hey jarvis") into
# ~/reactor/models and checks them against known hashes.
set -euo pipefail

DEST="${1:-$HOME/reactor/models}"
BASE="https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
mkdir -p "$DEST"

fetch() {
  local name="$1" sum="$2"
  if [ -f "$DEST/$name" ] && echo "$sum  $DEST/$name" | sha256sum -c --status; then
    echo "  have  $name"
    return
  fi
  curl -sSfL -o "$DEST/$name.part" "$BASE/$name"
  echo "$sum  $DEST/$name.part" | sha256sum -c --status || {
    echo "  BAD   $name - the download does not match; not using it"
    rm -f "$DEST/$name.part"
    exit 1
  }
  mv "$DEST/$name.part" "$DEST/$name"
  echo "  got   $name"
}

fetch melspectrogram.onnx ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f
fetch embedding_model.onnx 70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f
fetch hey_jarvis_v0.1.onnx 94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb
echo "models ready in $DEST"
