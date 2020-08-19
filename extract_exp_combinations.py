import os
import sys

PRJCT_FOLDER = '/scratch/gpfs/hgazula'
PARENT_FOLDER = 'brain2text-experiments'
ARG_STRINGS = [
    '--model', '--subjects', '--max-electrodes', '--window-size', '--shift',
    '--bin-size', '--weight-decay', '--tf-dropout', '--tf-nlayer',
    '--tf-nhead', '--tf-dmodel', '--tf-dff', '--temp', '--lr', '--gpus',
    '--epochs', '--batch-size', '--vocab-min-freq'
]
EXP_FOLDER = sys.argv[1]


def main():
    """Returns Experiment Configuration in a folder
    """
    full_file_path = os.path.join(PRJCT_FOLDER, PARENT_FOLDER, EXP_FOLDER)

    with open(os.path.join(full_file_path, 'folder_list'), 'r') as file_h:
        file_list = file_h.readlines()

    file_list = [item.rstrip().split('_') for item in file_list]
    exp_combination = zip(*file_list)

    for k, v in dict(zip(ARG_STRINGS, exp_combination)).items():
        print(k.lstrip('-'), ':', list(set(v)))


if __name__ == '__main__':
    main()
