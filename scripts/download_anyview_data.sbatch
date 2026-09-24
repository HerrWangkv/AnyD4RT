#!/bin/bash
#SBATCH --job-name=anyview-dl
#SBATCH --partition=dev_cpuonly
#SBATCH --account=hk-project-p0023969
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --constraint=LSDF
#SBATCH --output=logs/anyview-dl-%j.out
# Resumable: if the 4 h limit is hit, just resubmit (sbatch scripts/download_anyview_data.sbatch).
set -euo pipefail
export PYTHONUNBUFFERED=1
echo "host $(hostname), start $(date)"
curl -sfI --max-time 20 https://s3.us-east-1.amazonaws.com/tri-ml-public.s3.amazonaws.com/datasets/anyview/manifests/Kubric5D.jsonl >/dev/null \
    || { echo "no internet on compute node"; exit 1; }
ls /lsdf/kit/mrt/projects/d4rt-data >/dev/null
bash scripts/download_anyview_data.sh
echo "end $(date)"
