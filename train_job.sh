#!/bin/bash

#SBATCH -q c2
#SBATCH -p compute
#SBATCH --mem 128GB
#SBATCH -c 8
#SBATCH -t 10-00:00:00

#SBATCH --mail-user=im2821@nyu.edu      
#SBATCH --mail-type=BEGIN,END,FAIL        

#SBATCH -o output.txt
#SBATCH -e error.txt



source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
conda activate chehabEnv

cd RL
python -m fhe_rl --tokenizer_type dynamic train