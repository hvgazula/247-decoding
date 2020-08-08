import os

PRJCT_FOLDER = os.getcwd()
EXP_FOLDER = 'Experiments_Wednesday'
PARENT_DIR = os.path.join(PRJCT_FOLDER, EXP_FOLDER)

folder_list_file = os.path.join(PARENT_DIR, 'folder_list')

with open(folder_list_file, 'r') as file_h:
    folder_list = file_h.readlines()

folder_list = [item.rstrip() for item in folder_list]

for folder in folder_list:
    full_folder_path = os.path.join(PARENT_DIR, folder)
    trials = os.listdir(full_folder_path)
    for trial in trials:
        with open(os.path.join(full_folder_path, trial, 'output'), 'r') as file_h: 
            lines = file_h.readlines()

    print(lines[-1])
