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
import sys
import pandas as pd
from nltk.stem import PorterStemmer

from arg_parser import arg_parser
from build_matrices import build_design_matrices
from config import build_config
from sklearn.model_selection import StratifiedKFold

def create_dict(*args):
    print('hello')
    print(globals().keys())
    return dict(((k, globals()[k]) for k in args))


def save_pickle(item, file_name):
    if '.pkl' not in file_name:
        file_name = file_name + '.pkl'

    with open(file_name, 'wb') as fh:
        pickle.dump(item, fh)
    return


def main():
    args = arg_parser()
    CONFIG = build_config(args, results_str='test')

    if CONFIG['pickle']:
        (full_signal, full_stitch_index, trimmed_signal, trimmed_stitch_index,
         binned_signal, bin_stitch_index, labels, convo_example_size,
         electrodes) = build_design_matrices(CONFIG,
                                             delimiter=" ",
                                             aug_shift_ms=[-1000, -500])

        full_signal_dict = dict(full_signal=full_signal,
                                full_stitch_index=full_stitch_index,
                                electrodes=electrodes)
        trimmed_signal_dict = dict(trimmed_signal=trimmed_signal,
                                   trimmed_stitch_index=trimmed_stitch_index,
                                   electrodes=electrodes)
        binned_signal_dict = dict(full_signal=binned_signal,
                                  full_stitch_index=bin_stitch_index,
                                  electrodes=electrodes)

        save_pickle(full_signal_dict, '625_full_signal')
        save_pickle(trimmed_signal_dict, '625_trimmed_signal')
        save_pickle(binned_signal_dict, '625_binned_signal')

        trimmed_stitch_index.insert(0, 0)
        trimmed_stitch_index.pop(-1)

        new_labels = []
        ps = PorterStemmer()
        for start, sub_list in zip(trimmed_stitch_index, labels):
            modified_labels = [(ps.stem(*i[0]), i[1], i[2] + start,
                                i[3] + start, i[4]) for i in sub_list]
            new_labels.extend(modified_labels)

        df = pd.DataFrame(
            new_labels,
            columns=['word', 'speaker', 'onset', 'offset', 'accuracy'])

        labels_dict = dict(labels=df.to_dict('records'),
                           convo_label_size=convo_example_size)

        save_pickle(labels_dict, '625_labels')

        # create and save folds
        df = df.groupby('word').filter(lambda x: len(x) >= args.vocab_min_freq)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        # Extract only test folds
        folds = [t[1] for t in skf.split(df, df.word)]

        label_folds = {}
        for index, fold in enumerate(folds):
            label_folds['fold' + str(index)] = df.iloc[fold].to_dict('records')
        
        save_pickle(label_folds, '625_label_folds')
    return


if __name__ == "__main__":
    start_time = datetime.now()
    print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

    main()

    end_time = datetime.now()
    print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
    print(f'Total runtime: {end_time - start_time} (HH:MM:SS)')
