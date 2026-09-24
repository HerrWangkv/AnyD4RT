#!/usr/bin/env bash
# Download AnyView-DVS checkpoints, AnyViewBench and Kubric-5D to LSDF (next to the D4RT data),
# verify sha256, extract, and symlink into third_party/AnyView-DVS/{checkpoints,data}.
# Resumable: rerun to continue; verified files and finished extractions are skipped.
set -euo pipefail
ROOT=${ANYVIEW_ROOT:-/lsdf/kit/mrt/projects/d4rt-data/anyview}
REPO=$(cd "$(dirname "$0")/.." && pwd)
AV=$REPO/third_party/AnyView-DVS
B=https://s3.us-east-1.amazonaws.com/tri-ml-public.s3.amazonaws.com/datasets/anyview
mkdir -p "$ROOT"/{archives,checkpoints,data}

# 1) checkpoints (4.5 GB) and 2) Kubric-5D archives (216 GB): manifest-based, parallel, resumable
python3 "$AV/scripts/download.py" --tier checkpoints --out "$ROOT/checkpoints"
python3 "$AV/scripts/download.py" --tier Kubric5D --out "$ROOT/archives" --workers 4

# 3) single-file archives: AnyViewBench (7 GB) + Kubric5D_tiny (2.2 GB)
cd "$ROOT/archives"
for f in AnyViewBench_zeroshot.tar.gz AnyViewBench_indist.tar.gz Kubric5D_tiny.tar.gz; do
    curl -fsSLO "$B/$f.sha256"
    sha256sum --status -c "$f.sha256" 2>/dev/null && { echo "ok $f"; continue; }
    curl -fSL -C - -o "$f" "$B/$f"
    sha256sum -c "$f.sha256"
done

# 4) extract every archive once (marker file per archive)
for f in "$ROOT"/archives/*.tar.gz; do
    m="$ROOT/data/.extracted_$(basename "$f")"
    [ -e "$m" ] && continue
    echo "extracting $(basename "$f")"
    tar xzf "$f" -C "$ROOT/data" && touch "$m"
done

# 5) link into the AnyView checkout
ln -sfn "$ROOT/checkpoints" "$AV/checkpoints"
ln -sfn "$ROOT/data" "$AV/data"
echo "done: $(du -sh "$ROOT" | cut -f1) in $ROOT"
