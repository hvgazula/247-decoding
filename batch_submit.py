import os
from itertools import product


def contains_exclude_dict(superitem, exclude):
    for subitem in exclude:
        if all(item in superitem.items() for item in subitem.items()):
            return 1
    return 0


def create_exclude_dicts():
    exclude_args = ['--subjects', '--max-electrodes']
    exclude_vals = [(625, 64), (676, 55)]

    exclude_dicts = [dict(zip(exclude_args, item)) for item in exclude_vals]
    return exclude_dicts


def create_script(job_name_str, s_list):
    file_name = os.path.join(os.getcwd(), 'slurm_scripts',
                             f'{job_name_str}.sh')
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, 'w+') as fh:
        fh.write("#!/bin/bash\n")
        fh.write(f"#SBATCH --job-name={job_name_str}\n")
        fh.write("#SBATCH --output=./slurm_logs/%j-%x.out\n")
        fh.write("#SBATCH --error=./slurm_logs/%j-%x.err\n")
        fh.write("#SBATCH --nodes=1 #nodes\n")
        fh.write("#SBATCH --ntasks-per-node=1\n")
        fh.write("#SBATCH --cpus-per-task=4\n")
        fh.write("#SBATCH --mem=16G\n")
        fh.write("#SBATCH --time=0-02:00:00\n")
        fh.write("#SBATCH --gres=gpu:1\n")
        fh.write("#SBATCH --mail-type=begin\n")
        fh.write("#SBATCH --mail-type=fail\n")
        fh.write("#SBATCH --mail-type=end\n")
        fh.write("#SBATCH --mail-user=hvgazula@umich.edu\n")
        fh.write("\n")
        fh.write("module purge\n")
        fh.write("module load anaconda3\n")
        fh.write("conda activate torch-env\n")
        fh.write("\n")
        fh.write("if [[ -v $SLURM_TASK_ARRAY_ID ]]\n")
        fh.write("then\n")
        fh.write("\tSEED=$SLURM_TASK_ARRAY_ID\n")
        fh.write("else\n")
        fh.write("\tSEED=1234\n")
        fh.write("fi\n")
        fh.write(f"python {os.path.join(os.getcwd(), 'main.py')} \\\n")
        for item in s_list:
            fh.write(f'\t{item} \\\n')
        fh.write("\t--seed $SEED \\\n")
        fh.write(f"\t--output-folder {job_name_str}\n")

    return file_name


# model = ["MeNTAL"]
# subjects = [625, 676]
# max_electrodes = [55, 64]
# shift = [25, 50, 100, 150, 250, 500]
# lr = 1e-4
# gpus = 1
# epochs = 100
# batch_size = 120
# bin_size = [25, 50, 75, 100, 125, 150]
# tf_weight_decay = [0.01, 0.025, 0.05, 0.075]
# tf_dropout = [0.1]
# tf_nlayers = [3, 6, 9, 12]
# tf_nheads = [4, 8, 16]
# tf_dmodel = [128, 256, 512, 1024]
# tf_dff = [128, 256, 512, 1024]
# temp = [0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]

model = ["MeNTAL"]
subjects = [625, 676]
max_electrodes = [55, 64]
shift = [0]
bin_size = [50]
tf_weight_decay = [0.01]
tf_dropout = tf_weight_decay
tf_nlayer = [3]
tf_nhead = [4]
tf_dmodel = [128]
tf_dff = [128]
temp = [0.9]
lr = [1e-4]
gpus = [2]
epochs = [100]
batch_size = [120]

arg_values = [
    model, subjects, max_electrodes, shift, bin_size, tf_weight_decay,
    tf_dropout, tf_nlayer, tf_nhead, temp, lr, gpus, epochs, batch_size
]

arg_strings = [
    '--model', '--subjects', '--max-electrodes', '--shift', '--bin-size',
    '--weight-decay', '--tf-dropout', '--tf-nlayer', '--tf-nhead', '--temp',
    '--lr', '--gpus', '--epochs', '--batch-size'
]

args_dict = dict(zip(arg_strings, arg_values))
a = product(*args_dict.values())

count = 0
for element in a:
    element_dict = dict(zip(arg_strings, element))

    if contains_exclude_dict(element_dict, create_exclude_dicts()):
        continue
    final_s = [' '.join(str(f) for f in tup) for tup in element_dict.items()]
    job_name_str = '_'.join([str(item) for item in element])
    file_name = create_script(job_name_str, final_s)
    os.system(f'sbatch {file_name}')
    count = count + 1

print(f"Number of slurm scripts generated is: {count}")
