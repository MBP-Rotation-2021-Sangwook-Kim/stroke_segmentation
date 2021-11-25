#!/bin/bash
#SBATCH --account=rrg-mgoubran
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64000M
#SBATCH --time=54:00:0
#SBATCH --out=/scratch/lboone/deepbench/experiments/networks/2/wmh_hab/models/unetr_f/%j.out

# when tested on CC, avg step time for BS=2 (i.e. 2 samples/step) is ~0.78 s

module load python cuda cudnn

# prepare virtualenv
virtualenv --no-download $SLURM_TMPDIR/monai-env
source $SLURM_TMPDIR/monai-env/bin/activate
# pip install git+https://github.com/lyndonboone/MONAI.git@db
pip install monai-weekly
pip install git+https://github.com/lyndonboone/torchio.git@deepbench
pip install --no-index nibabel
pip install --no-index matplotlib
pip install --no-index tqdm
pip install --no-index scipy
pip install --no-index pandas
pip install einops

# prepare data
mkdir $SLURM_TMPDIR/data
tar -xf /scratch/lboone/deepbench/datasets/wmh_hab.tar.gz -C $SLURM_TMPDIR/data

# prepare tmp output dir
mkdir $SLURM_TMPDIR/out

# start training
python /scratch/lboone/deepbench/experiments/networks/2/wmh_hab/models/unetr_f/train.py

# copy output files
cp -r $SLURM_TMPDIR/out /scratch/lboone/deepbench/experiments/networks/2/wmh_hab/models/unetr_f
