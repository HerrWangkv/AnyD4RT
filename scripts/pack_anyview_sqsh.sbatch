#!/bin/bash
#SBATCH --job-name=anyview-sqsh
#SBATCH --partition=dev_cpuonly
#SBATCH --account=hk-project-p0023969
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=76
#SBATCH --time=04:00:00
#SBATCH --constraint=LSDF
#SBATCH --output=logs/anyview-sqsh-%j.out
# Pack each extracted AnyView dataset into one .sqsh (zstd, 128K blocks, same as the d4rt/*.sqsh),
# then check the file count inside each image against the source directory.
# Skips images that already exist; mount with: squashfuse <name>.sqsh <mountpoint>
set -euo pipefail
ROOT=/lsdf/kit/mrt/projects/d4rt-data/anyview
declare -A OUT=(
    [Kubric5D_train]=kubric5d_train [Kubric5D_val]=kubric5d_val [Kubric5D_test]=kubric5d_test
    [Kubric5D_tiny]=kubric5d_tiny [AnyViewBench_zeroshot]=anyviewbench_zeroshot
    [AnyViewBench_indist]=anyviewbench_indist
)
for d in "${!OUT[@]}"; do
    src=$ROOT/data/$d; dst=$ROOT/${OUT[$d]}.sqsh
    [ -e "$dst" ] && { echo "skip $dst (exists)"; continue; }
    echo "[$(date +%T)] packing $d"
    mksquashfs "$src" "$dst.tmp" -comp zstd -b 131072 -no-xattrs -processors "$SLURM_CPUS_PER_TASK" -no-progress -quiet
    n_src=$(find "$src" -type f | wc -l)
    n_img=$(unsquashfs -lls "$dst.tmp" | grep -c '^-')
    [ "$n_src" = "$n_img" ] || { echo "MISMATCH $d: $n_src files vs $n_img in image"; exit 1; }
    mv "$dst.tmp" "$dst"
    echo "[$(date +%T)] ok $d: $n_img files, $(du -h "$dst" | cut -f1)"
done
echo "all images built"
