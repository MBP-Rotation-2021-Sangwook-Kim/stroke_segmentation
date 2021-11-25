#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=2  # Refer to cluster's documentation for the right CPU/GPU ratio
#SBATCH --mem=1024M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-03:00     # DD-HH:MM:SS
#SBATCH --

module load python/3.6 cuda cudnn

# ENV VARIABLES
SOURCEDIR=/scratch/sangwook/projects/stroke_segmentation

# Prepare virtualenv
source ~/torch38/bin/activate
# You could also create your environment here, on the local storage ($SLURM_TMPDIR), for better performance. See our docs on virtual environments.

# Prepare data
# mkdir $SLURM_TMPDIR/data
# tar xf ~/projects/def-xxxx/data.tar -C $SLURM_TMPDIR/data

# Start training
python $SOURCEDIR/health.py