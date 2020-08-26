import os
import sys

PRJCT_FOLDER = os.getcwd()
EXP_FOLDER = sys.argv[1]
PARENT_DIR = os.path.join(PRJCT_FOLDER, EXP_FOLDER)


def return_folders():
    folder_list_file = os.path.join(PARENT_DIR, 'folder_list')

    with open(folder_list_file, 'r') as file_h:
        folder_list = file_h.readlines()

    return [item.rstrip() for item in folder_list]


def read_output(full_folder_path, trial):
    output_file = os.path.join(full_folder_path, trial, 'output')
    with open(output_file, 'r') as file_h:
        lines = file_h.readlines()
    return lines[-1]


def extract_time(time_line):
    if 'runtime' in time_line:
        print(time_line.strip().split(' ')[2])


def main():
    folder_list = return_folders()
    for folder in folder_list:
        full_folder_path = os.path.join(PARENT_DIR, folder)
        try:
            trials = os.listdir(full_folder_path)
            for trial in trials:
                time_line = read_output(full_folder_path, trial)
                extract_time(time_line)
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    main()
