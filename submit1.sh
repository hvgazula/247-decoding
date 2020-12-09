#!/bin/bash
#SBATCH --time=04:30:00
#SBATCH --mem=4GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH -o './logs/%A-%a.out'
#SBATCH -e './logs/%A.err'

if [[ "$HOSTNAME" == *"tiger"* ]]
then
    echo "It's tiger"
    module load anaconda
    source activate 247-podcast-tf2
else
    module load anacondapy
    source activate srm
fi

echo 'Requester:' $USER
echo 'Node:' $HOSTNAME
echo 'Start time:' `date`
echo "$@"
if [[ -v SLURM_ARRAY_TASK_ID ]]
then
    python "$@" --electrodes $SLURM_ARRAY_TASK_ID
else
    python tfsdec_main.py
fi
echo 'End time:' `date`
