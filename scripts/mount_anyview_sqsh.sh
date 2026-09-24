#!/usr/bin/env bash
# Mount the AnyView squashfs images read-only under one directory, laid out the way
# third_party/AnyView-DVS/data expects (Kubric5D_train/, AnyViewBench_zeroshot/, ...).
# Mounts are per node: run this on every node (and inside every sbatch job) that reads the data.
#   D4RT_DATA_ROOT  d4rt-data root (default: LSDF; on mrtknecht3: /tmp/kwang-data)
#   ANYVIEW_MNT     mount directory (default: /tmp/$USER/anyview-data)
# Usage: scripts/mount_anyview_sqsh.sh [--unmount]
set -euo pipefail
ROOT=${D4RT_DATA_ROOT:-/lsdf/kit/mrt/projects/d4rt-data}
MNT=${ANYVIEW_MNT:-/tmp/$USER/anyview-data}
declare -A IMAGES=(
    [Kubric5D_train]=kubric5d_train [Kubric5D_val]=kubric5d_val [Kubric5D_test]=kubric5d_test
    [Kubric5D_tiny]=kubric5d_tiny [AnyViewBench_zeroshot]=anyviewbench_zeroshot
    [AnyViewBench_indist]=anyviewbench_indist
)
for name in "${!IMAGES[@]}"; do
    dir=$MNT/$name
    if [ "${1:-}" = --unmount ]; then
        mountpoint -q "$dir" && fusermount -u "$dir" && echo "unmounted $dir"
        continue
    fi
    mkdir -p "$dir"
    mountpoint -q "$dir" && continue
    squashfuse "$ROOT/anyview/${IMAGES[$name]}.sqsh" "$dir"
    echo "mounted $dir"
done
