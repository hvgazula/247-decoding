import argparse
import copy
import os
from itertools import product

import numpy as np


def experiment_folder(args):
    return os.path.join(os.getcwd(), args.experiment_suffix)


def contains_exclude_dict(superitem, exclude):
    for subitem in exclude:
        if all(item in superitem.items() for item in subitem.items()):
            return 1
    return 0


def gather_results_folders(args, folder_list):
    output_file = os.path.join(experiment_folder(args), 'folder_list')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(output_file):
        with open(output_file, 'r') as file_h:
            prev_lines = file_h.readlines()
        prev_lines = [item.rstrip() for item in prev_lines]

        folder_list = sorted(set(folder_list) - set(prev_lines))

    with open(output_file, 'a+') as file_h:
        file_h.writelines(folder + '\n' for folder in folder_list)


def create_exclude_dicts():
    exclude_args = ['--subjects', '--max-electrodes']
    exclude_vals = [(625, 64), (676, 55)]

    exclude_dicts = [dict(zip(exclude_args, item)) for item in exclude_vals]
    return exclude_dicts


def create_script(job_name_str, s_list, args):
    file_name = os.path.join(experiment_folder(args), 'slurm_scripts',
                             f'{job_name_str}.sh')
    slurm_logs_folder = os.path.join(experiment_folder(args), 'slurm_logs')

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    os.makedirs(slurm_logs_folder, exist_ok=True)

    with open(file_name, 'w+') as fh:
        fh.write("#!/bin/bash\n")
        fh.write(f"#SBATCH --job-name={job_name_str}\n")
        fh.write(f"#SBATCH --output={slurm_logs_folder}/%j-%x.out\n")
        fh.write(f"#SBATCH --error={slurm_logs_folder}/%j-%x.err\n")
        fh.write("#SBATCH --nodes=1 #nodes\n")
        fh.write("#SBATCH --ntasks-per-node=1\n")
        fh.write("#SBATCH --cpus-per-task=2\n")
        fh.write("#SBATCH --mem=16G\n")
        fh.write(f"#SBATCH --time={ALLOCATE_TIME}\n")
        fh.write(f"#SBATCH --gres=gpu:{ALLOCATE_GPUS}\n")
        fh.write("##SBATCH --mail-type=begin\n")
        fh.write("#SBATCH --mail-type=fail\n")
        fh.write("##SBATCH --mail-type=end\n")
        fh.write("#SBATCH --mail-user=hvgazula@umich.edu\n")
        fh.write("\n")
        fh.write("module purge\n")
        fh.write("module load anaconda3\n")
        fh.write("conda activate torch-env\n")
        fh.write("\n")
        fh.write("if [[ -v SLURM_ARRAY_TASK_ID ]]\n")
        fh.write("then\n")
        fh.write("\tSEED=$SLURM_ARRAY_TASK_ID\n")
        fh.write("else\n")
        fh.write("\tSEED=1234\n")
        fh.write("fi\n\n")
        fh.write(f"python {os.path.join(os.getcwd(), 'main.py')} \\\n")
        for item in s_list:
            fh.write(f'\t{item} \\\n')
        if NGRAM_FLAG:
            fh.write("\t--ngrams \\\n")
        if NSEQ_FLAG:
            fh.write("\t--nseq \\\n")
        fh.write("\t--seed $SEED \\\n")
        fh.write(f"\t--output-folder {job_name_str} \\\n")
        fh.write(f"\t--exp-suffix {args.experiment_suffix}\n")
    return file_name


ALLOCATE_GPUS = 1
ALLOCATE_TIME = 105
NGRAM_FLAG = 1
NSEQ_FLAG = 0
MAX_JOBS = 5


def experiment_configuration():
    model = ["PITOM", "MeNTALmini"]
    subjects = [676]
    max_electrodes = [55, 64]
    window_size = [['175', '175'], ['175', '250']]
    shift = [0]
    bin_size = [50]
    tf_weight_decay = [0.01]
    tf_dropout = tf_weight_decay
    tf_nlayer = [3]
    tf_nhead = [4]
    tf_dmodel = [512]
    tf_dff = [1024]
    temp = [0.9]
    lr = [1e-4]
    gpus = [2]
    epochs = [100]
    batch_size = [240]
    vocab_min_freq = [20, 40, 60]

    arg_values = [
        model, subjects, max_electrodes, window_size, shift, bin_size,
        tf_weight_decay, tf_dropout, tf_nlayer, tf_nhead, tf_dmodel, tf_dff,
        temp, lr, gpus, epochs, batch_size, vocab_min_freq
    ]

    arg_strings = [
        '--model', '--subjects', '--max-electrodes', '--window-size',
        '--shift', '--bin-size', '--weight-decay', '--tf-dropout',
        '--tf-nlayer', '--tf-nhead', '--tf-dmodel', '--tf-dff', '--temp',
        '--lr', '--gpus', '--epochs', '--batch-size', '--vocab-min-freq'
    ]

    return arg_values, arg_strings


def already_exists(args, job_string):
    output_file = os.path.join(experiment_folder(args), 'folder_list')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not os.path.exists(output_file):
        return False

    with open(output_file, 'r') as file_h:
        prev_lines = file_h.readlines()
    prev_lines = [item.rstrip() for item in prev_lines]

    return job_string in prev_lines


def main(args):
    arg_values, arg_strings = experiment_configuration()
    args_dict = dict(zip(arg_strings, arg_values))
    a = product(*args_dict.values())

    results_folders = []
    for element in a:
        element_dict = dict(zip(arg_strings, element))
        element_dict1 = copy.deepcopy(element_dict)
        for k, v in element_dict.items():
            if isinstance(v, list):
                element_dict[k] = ' '.join([str(elem) for elem in v])
                element_dict1[k] = ''.join([str(elem) for elem in v])

        if contains_exclude_dict(element_dict, create_exclude_dicts()):
            continue
        final_s = [
            ' '.join(str(f) for f in tup) for tup in element_dict.items()
        ]
        job_name_str1 = '_'.join(
            [str(value) for value in element_dict1.values()])

        if not already_exists(args, job_name_str1):
            file_name = create_script(job_name_str1, final_s, args)
            if not os.system(f'sbatch --array=01-{MAX_JOBS} {file_name}'):
                results_folders.append(job_name_str1)

    gather_results_folders(args, results_folders)

    print(f"Number of slurm scripts generated is: {len(results_folders)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_suffix')
    args = parser.parse_args()

    main(args)
