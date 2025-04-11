#!/bin/bash

#SBATCH --mem=8G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --job-name=llm_training
#SBATCH --output=llm_training%j.out
#SBATCH --mail-user=<vbertalan@gmail.com>
#SBATCH --mail-type=ALL

echo "GPU available on host:"
nvidia-smi

echo "Acessing folder..."
cd /home/vberta/projects/def-aloise/vberta/
source /home/vberta/projects/def-aloise/vberta/vbertapy/bin/activate

echo "Running code..."
python /home/vberta/projects/def-aloise/vberta/Paper3/llm-trainer-mini.py