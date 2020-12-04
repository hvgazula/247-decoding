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

import pandas as pd
from pprint import pprint

from arg_parser import arg_parser
from build_matrices import build_design_matrices
from config import build_config


def main():
    args = arg_parser()
    CONFIG = build_config(args, results_str='test')

    if CONFIG['pickle']:
        (full_signal, full_stitch_index, trimmed_signal, trimmed_stitch_index,
         binned_signal, bin_stitch_index,
         labels) = build_design_matrices(CONFIG,
                                         delimiter=" ",
                                         aug_shift_ms=[-1000, -500])

        full_signal_dict = dict(full_signal=full_signal,
                                full_stitch_index=full_stitch_index)
        binned_signal_dict = dict(binned_signal=binned_signal,
                                  bin_stitch_index=bin_stitch_index)

        with open('625_full_signal.pkl', 'wb') as fh:
            pickle.dump(full_signal_dict, fh)

        with open('625_trimmed_signal.pkl', 'wb') as fh:
            pickle.dump(full_signal_dict, fh)

        with open('625_binned_signal.pkl', 'wb') as fh:
            pickle.dump(binned_signal_dict, fh)

        full_stitch_index.insert(0, 0)
        full_stitch_index.pop(-1)

        new_labels = []
        for start, sub_list in zip(full_stitch_index, labels):
            modified_labels = [(*i[0], i[1], i[2] + start, i[3] + start, i[4])
                               for i in sub_list]
            new_labels.extend(modified_labels)


        df = pd.DataFrame(
            new_labels[:10],
            columns=['word', 'speaker', 'onset', 'offset', 'accuracy'])

        print(df.to_dict('records'))

        with open('625_labels.pkl', 'wb') as fh:
            pickle.dump(df.to_dict('records'), fh)

    return


if __name__ == "__main__":
    start_time = datetime.now()
    print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

    main()

    end_time = datetime.now()
    print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
    print(f'Total runtime: {end_time - start_time} (HH:MM:SS)')
