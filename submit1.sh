#!/bin/bash
#SBATCH --time=01:10:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -o './logs/%A-%a.log'

if [[ "$HOSTNAME" == *"tiger"* ]]
then
    module load anaconda
    conda activate 247-podcast-tf2
else
    module load anacondapy
    source activate 247-main
fi

echo 'Requester:' $USER
echo 'Node:' $HOSTNAME
echo 'Start time:' `date`
echo "$@"
for run in {0..4}; do
    python "$@"
done
echo 'End time:' `date`
