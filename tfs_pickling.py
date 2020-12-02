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

from arg_parser import arg_parser
from build_matrices import build_design_matrices
from config import build_config


def main():
    args = arg_parser()
    CONFIG = build_config(args, results_str='test')

    if CONFIG['pickle']:
        pickle_output = build_design_matrices(CONFIG,
                                              delimiter=" ",
                                              aug_shift_ms=[-1000, -500])

        keys = [
            'full_signal', 'full_stitch_index', 'binned_signal',
            'bin_stitch_index'
        ]
        records = dict(zip(keys, pickle_output))

        with open('625_data_pickle', 'wb') as fh:
            pickle.dump(records, fh)

    return


if __name__ == "__main__":
    start_time = datetime.now()
    print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

    main()

    end_time = datetime.now()
    print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
    print(f'Total runtime: {end_time - start_time} (HH:MM:SS)')
