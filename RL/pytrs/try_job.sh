#!/bin/bash

#SBATCH -q c2
#SBATCH -p nvidia
#SBATCH --gres=gpu:a100:1
#SBATCH --mem 64GB
#SBATCH -c 16
#SBATCH -t 5-00:00:00

#SBATCH --mail-user=im2821@nyu.edu        
#SBATCH --mail-type=BEGIN,END,FAIL        

#SBATCH -o output.txt
#SBATCH -e error.txt



source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehabEnv

cd RL
python -m fhe_rl --tokenizer_type dynamic train
