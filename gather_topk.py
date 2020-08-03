import json
import os
import re

import numpy as np
import pandas as pd


def extract_floats(expr):
    mylist = re.findall(r"[-+]?\d*\.\d+|\d+", expr)
    return [float(item) for item in mylist]


def flatten_list(big_list):
    return [item for sublist in big_list for item in sublist]


def average_trials(item):
    array = np.array(item)
    mean_array = np.mean(array, axis=0)
    return mean_array.tolist()


arg_strings = [
    'model', 'subjects', 'max-electrodes', 'window-size', 'shift', 'bin-size',
    'weight-decay', 'tf-dropout', 'tf-nlayer', 'tf-nhead', 'tf-dmodel',
    'tf-dff', 'temp', 'lr', 'gpus', 'epochs', 'batch-size'
]

topk_cols = [
    'top1', 'top1-chance', 'top5', 'top5-chance', 'top10', 'top10-chance'
]

PRJCT_FOLDER = os.getcwd()
EXP_FOLDER = 'TestRuns'
PARENT_DIR = os.path.join(PRJCT_FOLDER, EXP_FOLDER)
TOPK_FILE_NAME = 'topk-val.txt'
AUC_SUMMARY = 'auc-summary-'
folder_list_file = os.path.join(PARENT_DIR, 'folder_list')

with open(folder_list_file, 'r') as file_h:
    folder_list = file_h.readlines()

folder_list = [item.rstrip() for item in folder_list]

dict_list = []
big_abc, big_auc = [], []
for folder in folder_list:
    exp_config = folder.split('_')
    dict_list.append(dict(zip(arg_strings, exp_config)))
    abc, jkl = [], []
    full_folder_path = os.path.join(PARENT_DIR, folder)
    trials = os.listdir(full_folder_path)
    for trial in trials:
        topk_file_path = os.path.join(full_folder_path, trial, TOPK_FILE_NAME)
        auc_file_path = os.path.join(full_folder_path, trial, AUC_SUMMARY)
        with open(topk_file_path, 'r') as file_h:
            topk_lines = file_h.readlines()[2:5]
        flat_list = flatten_list(
            [extract_floats(line.split('\t')[1]) for line in topk_lines])
        with open(auc_file_path, 'r') as file_h:
            avg_auc = json.load(file_h)['rocauc_w_avg']
        jkl.append(avg_auc)
        abc.append(flat_list)
    big_abc.append(average_trials(abc))
    big_auc.append(sum(jkl) / len(jkl))


topk_dict_list = [dict(zip(topk_cols, item)) for item in big_abc]
df1 = pd.DataFrame(dict_list)
df2 = pd.DataFrame(topk_dict_list)
df3 = pd.DataFrame(big_auc, columns=['auc_w_avg'])

df = pd.concat([df1, df2, df3], axis=1)
output_xls = os.path.join(PARENT_DIR, EXP_FOLDER + '.xlsx')

writer = pd.ExcelWriter(output_xls,
                        engine='xlsxwriter',
                        options={'strings_to_numbers': True})

df.to_excel(writer, index=False)
writer.save()
