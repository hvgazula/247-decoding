import os

import pandas as pd

from filter_utils import label_counts


def format_dataframe(df):
    for column in df.select_dtypes(include='object'):
        df[column] = df[column].map('{:8s}'.format)
    for column in df.select_dtypes(include='int'):
        df[column] = df[column].map('{:5d}'.format)
    df.columns = df.columns.map('{:8s}'.format)

    return df


def bigram_counts_to_csv(CONFIG, labels, classify=True, data_str=None):
    label_counter = label_counts(labels, classify=classify)

    col_size = 1 if classify else len(list(label_counter.keys())[0])

    col_names = [
        '_'.join(['word', str(num)]) for num in range(1, col_size + 1)
    ]
    df = pd.Series(label_counter).rename_axis(
        col_names).sort_index().reset_index(name='Count')
    df = format_dataframe(df)

    if not data_str:
        print('No file name specified.')
    elif data_str == 'mixed':
        file_name = '_'.join(['train_test', 'gram', 'count']) + '.csv'
    else:
        file_name = '_'.join([data_str, 'count']) + '.csv'

    df.to_csv(os.path.join(CONFIG["SAVE_DIR"], file_name), index=False)

    return None


def save_word_counter(CONFIG, word2freq):
    '''Save word counter'''
    print("Saving word counter")
    df = pd.Series(word2freq).rename_axis(['Word'
                                           ]).reset_index(name='Frequency')
    df = format_dataframe(df)
    df.to_csv(os.path.join(CONFIG["SAVE_DIR"], 'word2freq.csv'), index=False)
