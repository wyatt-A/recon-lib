#!/bin/bash
#SBATCH --job-name=prep_lr_filter
#SBATCH --output=prep_lr_%A_%a.out
#SBATCH --error=prep_lr_%A_%a.err
#SBATCH --array=0-149
#SBATCH --mem=32G

WORKDIR="/privateShares/wa41/26.wang.06/N61620"
srun ./prep_lr_reco filter "$WORKDIR" $SLURM_ARRAY_TASK_ID 512 256 256