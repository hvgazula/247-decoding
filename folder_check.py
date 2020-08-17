import os

SCRATCH_DIR = '/scratch/gpfs/hgazula'
PRJCT_DIR = 'brain2text-experiments'
EXP_FOLDER_NAME = 'experiments-pitom-676'

full_path = os.path.join(SCRATCH_DIR, PRJCT_DIR, EXP_FOLDER_NAME)
EXCLUDE_FOLDERS = ['folder_list', 'slurm_logs', 'slurm_scripts']

for folder in os.listdir(full_path):
    if folder in EXCLUDE_FOLDERS:
        continue
    sub_folder = os.path.join(full_path, folder)
    trial_list = os.listdir(sub_folder)
    if len(trial_list) != 5:
        print(folder)
        continue
    for trial in sorted(trial_list):
        curr_folder = os.path.join(sub_folder, trial)
        auc_list = [
            f for f in os.listdir(curr_folder) if f.startswith('auc-summary-')
        ]
        file_size = [
            os.stat(os.path.join(curr_folder, file)).st_size
            for file in auc_list
        ]
        if len(auc_list) != 3 or 0 in file_size:
            print(folder)
