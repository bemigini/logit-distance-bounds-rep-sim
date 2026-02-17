#!/bin/bash	

#SBATCH --job-name=resnetcifar100
#SBATCH --output=resnetcifar100_result-%J.out
#SBATCH --cpus-per-task=1
#SBATCH --time=95:00:00
#SBATCH --mem=14gb
#SBATCH --gres=gpu:1
#SBATCH --mail-user=user@mail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --export=ALL

## INFO
echo "Node: $(hostname)"
echo "Start: $(date +%F-%R:%S)"
echo -e "Working dir: $(pwd)\n"

conda init bash
source ~/.bashrc

source /opt/miniconda3/etc/profile.d/conda.sh

cd distillation-similarity/ 
conda activate rep-distill

echo -e "Working dir: $(pwd)\n"

python main.py --dataset cifar100 --model classifier --n_epochs 10 --lr 0.001 --batch_size 32 --exp_decay 0.995 --seed 10 --backbone pretrain_resnet50v2 --final_dim 50


echo "Done: $(date +%F-%R:%S)"