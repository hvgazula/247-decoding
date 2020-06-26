import os

import pandas as pd
from tabulate import tabulate

from filter_utils import label_counts


def bigram_counts_to_csv(CONFIG, labels_list, classify=True, data_str=None):
    classify = CONFIG["classify"]
    labels, y_train, y_test = labels_list
    all_labels_counter = label_counts(labels, classify=classify)
    train_labels_counter = label_counts(y_train, classify=classify)
    test_labels_counter = label_counts(y_test, classify=classify)

    col_size = 1 if classify else len(list(all_labels_counter.keys())[0])

    col_names = [
        '_'.join(['word', str(num)]) for num in range(1, col_size + 1)
    ]
    df_all = pd.Series(all_labels_counter).rename_axis(
        col_names).sort_index().reset_index(name='Total_Count')
    df_train = pd.Series(train_labels_counter).rename_axis(
        col_names).sort_index().reset_index(name='Train_Count')
    df_test = pd.Series(test_labels_counter).rename_axis(
        col_names).sort_index().reset_index(name='Test_Count')

    if not data_str:
        print('No file name specified.')
    elif data_str == 'mixed':
        file_name = '_'.join(['train_test', 'gram', 'count']) + '.csv'
    else:
        file_name = '_'.join([data_str, 'count']) + '.csv'

    df = pd.merge(df_train, df_test, on=['word_1', 'word_2'])
    df = pd.merge(df, df_all, on=['word_1', 'word_2'])

    tabulate_and_print(CONFIG, df, file_name)

    return None


def save_word_counter(CONFIG, word2freq):
    '''Save word counter'''
    print("Saving word counter")
    df = pd.Series(word2freq).rename_axis(['Word'
                                           ]).reset_index(name='Frequency')
    tabulate_and_print(CONFIG, df, 'word2freq.csv')


def print_model(CONFIG, model):
    print('Printing Model Summary')
    with open(os.path.join(CONFIG["SAVE_DIR"], 'model_summary'),
              'w') as file_h:
        print(model, file=file_h)


def tabulate_and_print(CONFIG, data_frame, file_name):
    mystrn = tabulate(data_frame,
                      headers=data_frame.columns,
                      showindex='False',
                      tablefmt='plain',
                      floatfmt=".4f",
                      numalign='center',
                      colalign=("center", ))

    with open(os.path.join(CONFIG["SAVE_DIR"], file_name), 'w') as f:
        f.writelines(mystrn)
