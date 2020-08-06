import json
import os
import re

import numpy as np
import pandas as pd

PRJCT_FOLDER = os.getcwd()
EXP_FOLDER = 'Experiments_Wednesday'
PARENT_DIR = os.path.join(PRJCT_FOLDER, EXP_FOLDER)


def extract_floats(expr):
    mylist = re.findall(r"[-+]?\d*\.\d+|\d+", expr)
    return [float(item) for item in mylist]


def flatten_list(big_list):
    return [item for sublist in big_list for item in sublist]


def average_trials(item):
    array = np.array(item)
    mean_array = np.mean(array, axis=0)
    return mean_array.tolist()


def whatever_function(word_type, topk_file, auc_summary):
    arg_strings = [
        'model', 'subjects', 'max-electrodes', 'window-size', 'shift',
        'bin-size', 'weight-decay', 'tf-dropout', 'tf-nlayer', 'tf-nhead',
        'tf-dmodel', 'tf-dff', 'temp', 'lr', 'gpus', 'epochs', 'batch-size'
    ]

    topk_cols = [
        'top1', 'top1-chance', 'top5', 'top5-chance', 'top10', 'top10-chance'
    ]

    topk_cols = ['-'.join([word_type, item]) for item in topk_cols]

    TOPK_FILE_NAME = topk_file
    AUC_SUMMARY = auc_summary
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
            topk_file_path = os.path.join(full_folder_path, trial,
                                          TOPK_FILE_NAME)
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
    df3 = pd.DataFrame(big_auc, columns=[word_type + '-auc_w_avg'])

    df = pd.concat([df2, df3], axis=1)

    return df, df1


df_bigram, df = whatever_function('bigram', 'topk-bigram.txt',
                              'auc-summary-bigram')
df_word1, _ = whatever_function('word1', 'topk-word1.txt', 'auc-summary-word1')
df_word2, _ = whatever_function('word2', 'topk-word2.txt', 'auc-summary-word2')


final_df = pd.concat([df, df_bigram, df_word1, df_word2], axis=1)

output_xls = os.path.join(PARENT_DIR, EXP_FOLDER + '.xlsx')

writer = pd.ExcelWriter(output_xls,
                        engine='xlsxwriter',
                        options={'strings_to_numbers': True})

final_df.to_excel(writer, index=False)
writer.save()
