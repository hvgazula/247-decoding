'''
Filename: /scratch/gpfs/hgazula/247-project/tfs_pickling.py
Path: /scratch/gpfs/hgazula/247-project
Created Date: Tuesday, December 1st 2020, 8:19:27 pm
Author: Harshvardhan Gazula
Description: Contains code to pickle 247 data

Copyright (c) 2020 Your Company
'''
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.model_selection import StratifiedKFold

from arg_parser import arg_parser
from build_matrices import build_design_matrices
from config import build_config


def save_pickle(item, file_name):
    """Write 'item' to 'file_name.pkl'
    """
    if '.pkl' not in file_name:
        file_name = file_name + '.pkl'

    with open(file_name, 'wb') as fh:
        pickle.dump(item, fh)
    return


def find_switch_points(array):
    return np.where(array[:-1] != array[1:])[0] + 1


def create_sentence(labels):
    """[summary]

    Args:
        labels ([type]): [description]

    Returns:
        [type]: [description]
    """
    convo_df = pd.DataFrame(
        labels, columns=['word', 'speaker', 'onset', 'offset', 'accuracy'])
    convo_df.word = convo_df['word'].apply(' '.join)

    # Find indices where speaker switches and split the dataframe
    speaker_switch_dx = find_switch_points(convo_df.speaker.values)
    sentence_df = np.split(convo_df, speaker_switch_dx, axis=0)

    # For each sentence df split
    my_labels = []
    for idx, section in enumerate(sentence_df):
        last_word_offset = section.iloc[-1, 3]
        first_word_onset = section.iloc[0, 2]
        sentence_len = last_word_offset - first_word_onset

        # create a sentence and append to the dataframe
        sentence = ' '.join(section['word'])
        sentence_arr = np.tile(sentence, (len(section), 1))
        sent_len_arr = np.tile(sentence_len, (len(section), 1))
        num_words = np.tile(len(section), (len(section), 1))
        sentence_idx = np.tile(idx + 1, (len(section), 1))

        my_labels.extend(
            np.concatenate(
                (section, sentence_arr, sent_len_arr, num_words, sentence_idx),
                axis=1).tolist())
    return my_labels


def add_sentences_to_labels(label_list):
    labels_with_sentences = []
    for convo in label_list:
        labels_with_sentence = create_sentence(convo)
        labels_with_sentences.append(labels_with_sentence)

    return labels_with_sentences


def adjust_label_onsets(trimmed_stitch_index, labels):
    """Adjust label onsets to account for stitched signal length.
    Also peform stemming on the labels.

    Args:
        trimmed_stitch_index (list): stitch indices of trimmed signal
        labels (list): of tuples (word, speaker, onset, offset, accuracy)

    Returns:
        DataFrame: labels
    """
    trimmed_stitch_index.insert(0, 0)
    trimmed_stitch_index.pop(-1)

    new_labels = []
    ps = PorterStemmer()

    len_to_add = 0
    for conv_id, (start,
                  sub_list) in enumerate(zip(trimmed_stitch_index, labels)):
        modified_labels = [
            (i[0], ps.stem(i[0]), i[1], i[2] + start, i[3] + start, i[4],
             *i[5:8], i[-1] + len_to_add, conv_id + 1) for i in sub_list
        ]
        new_labels.extend(modified_labels)
        len_to_add += sub_list[-1][-1]

    df = pd.DataFrame(np.vstack(new_labels),
                      columns=[
                          'word', 'stemmed_word', 'speaker', 'onset', 'offset',
                          'accuracy', 'sentence', 'sentence_length',
                          'words_in_sentence', 'sentence_idx',
                          'conversation_id'
                      ])

    return df


def create_label_pickles(args, df, file_string):
    """create and save folds

    Args:
        args (namespace): namespace object with input arguments
        df (DataFrame): labels
        file_string (str): output pickle name
    """
    df = df.groupby('word').filter(
        lambda x: len(x) >= args.vocab_min_freq).reset_index(drop=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    # Extract only test folds
    folds = [t[1] for t in skf.split(df, df.word)]

    # Go through each fold, and split
    for i in range(5):
        # Shift the number of folds for this iteration
        # [0 1 2 3 4] -> [1 2 3 4 0] -> [2 3 4 0 1]
        #                       ^ dev fold
        #                         ^ test fold
        #                 | - | <- train folds
        fold_col = 'fold' + str(i)
        folds_ixs = np.roll(range(5), i)
        *train_fold, dev_fold, test_fold = folds_ixs

        df.loc[folds[test_fold], fold_col] = 'test'
        df.loc[folds[dev_fold], fold_col] = 'dev'
        df.loc[[
            *folds[train_fold[0]], *folds[train_fold[1]], *folds[train_fold[2]]
        ], fold_col] = 'train'

    label_folds = df.to_dict('records')
    save_pickle(label_folds, file_string + str(args.vocab_min_freq))

    return


def main():
    args = arg_parser()
    CONFIG = build_config(args, results_str='pickles_new')

    if CONFIG['pickle']:
        (full_signal, full_stitch_index, trimmed_signal, trimmed_stitch_index,
         binned_signal, bin_stitch_index, labels, convo_example_size,
         electrodes) = build_design_matrices(CONFIG, delimiter=" ")

        # # Create pickle with full signal
        # full_signal_dict = dict(full_signal=full_signal,
        #                         full_stitch_index=full_stitch_index,
        #                         electrodes=electrodes)
        # save_pickle(full_signal_dict, '625_full_signal')

        # # Create pickle with trimmed signal
        # trimmed_signal_dict = dict(trimmed_signal=trimmed_signal,
        #                            trimmed_stitch_index=trimmed_stitch_index,
        #                            electrodes=electrodes)
        # save_pickle(trimmed_signal_dict, '625_trimmed_signal')

        # # Create pickle with binned signal
        # binned_signal_dict = dict(binned_signal=binned_signal,
        #                           bin_stitch_index=bin_stitch_index,
        #                           electrodes=electrodes)
        # save_pickle(binned_signal_dict, '625_binned_signal')

        # Create pickle with all labels
        labels = add_sentences_to_labels(labels)
        labels_df = adjust_label_onsets(trimmed_stitch_index, labels)
        labels_dict = dict(labels=labels_df.to_dict('records'),
                           convo_label_size=convo_example_size)
        save_pickle(labels_dict, '676_all_labels')

        # Create pickle with both production & comprehension labels
        create_label_pickles(args, labels_df, '625_both_labels_MWF')

        # Create pickle with production labels
        prod_df = labels_df[labels_df['speaker'] == 'Speaker1']
        create_label_pickles(args, prod_df, '676_prod_labels_MWF')

        # Create pickle with comprehension labels
        comp_df = labels_df[labels_df['speaker'] != 'Speaker1']
        create_label_pickles(args, comp_df, '676_comp_labels_MWF')

    return


if __name__ == "__main__":
    start_time = datetime.now()
    print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

    main()

    end_time = datetime.now()
    print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
    print(f'Total runtime: {end_time - start_time} (HH:MM:SS)')
