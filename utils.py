import math
import os
import random

import numpy as np
import pandas as pd
import torch


def read_file(fh):
    """Read file line-by-line and store in list

    Args:
        fh (string): name of the file

    Returns:
        list: contents of the file
    """
    with open(fh, 'r') as f:
        lines = [line.rstrip() for line in f]
    print(f'Number of Conversations is: {len(lines)}')
    return lines


def fix_random_seed(CONFIG):
    """Fix random seed
    """
    SEED = CONFIG['seed']
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def return_conversations(CONFIG):
    """Returns list of conversations

    Arguments:
        CONFIG {dict} -- Configuration information
        set_str {string} -- string indicating set type (train or valid)

    Returns:
        list -- List of tuples (directory, file, idx)
    """
    conversations = []

    for conv_dir in CONFIG["CONV_DIRS"]:
        conversation_list = os.listdir(conv_dir)
        conversations.append(conversation_list)

    convs = [
        (conv_dir + conv_name, '/misc/*datum_%s.txt' % ds, idx, electrode_list)
        for idx, (conv_dir, convs, ds, electrode_list) in enumerate(
            zip(CONFIG["CONV_DIRS"], conversations, CONFIG["datum_suffix"],
                CONFIG["electrode_list"])) for conv_name in convs
    ]

    return convs


def return_examples(file, delim, ex_words, vocab_str='std'):
    with open(file, 'r') as fin:
        lines = map(lambda x: x.split(delim), fin)
        examples = map(
            lambda x: (" ".join([
                z for y in x[0:-4]
                if (z := y.lower().strip().replace('"', '')) not in ex_words
            ]), x[-1].strip() == "Speaker1", x[-4], x[-3]), lines)
        examples = filter(lambda x: len(x[0]) > 0, examples)
        if vocab_str == 'spm':
            examples = map(
                lambda x:
                (vocabulary.EncodeAsIds(x[0]), x[1], int(float(x[2])),
                 int(float(x[3]))), examples)
        elif vocab_str == 'std':
            examples = map(
                lambda x:
                (x[0].split(), x[1], int(float(x[2])), int(float(x[3]))),
                examples)
        else:
            print("Bad vocabulary string")
        return list(examples)


def calculate_windows_params(gram, param_dict):
    seq_length = gram[3] - gram[2]
    begin_window = gram[2] + param_dict['start_offset']
    end_window = gram[3] + param_dict['end_offset']
    bin_size = int(
        math.ceil((end_window - begin_window) /
                  param_dict['bin_fs']))  # calculate number of bins

    return seq_length, begin_window, end_window, bin_size


def convert_ms_to_fs(CONFIG, fs=512):
    """Convert seconds to frames
    Arguments:
        CONFIG {dict} -- Configuration information
    Keyword Arguments:
        fs {int} -- Frames per second (default: {512})
    """
    window_ms = CONFIG["window_size"]
    shift_ms = CONFIG["shift"]
    bin_ms = CONFIG["bin_size"]

    bin_fs = int(bin_ms / 1000 * fs)
    shift_fs = int(shift_ms / 1000 * fs)
    window_fs = int(window_ms / 1000 * fs)
    half_window = window_fs // 2
    start_offset = -half_window + shift_fs
    end_offset = half_window + shift_fs

    signal_param_dict = dict()
    signal_param_dict['bin_fs'] = bin_fs
    signal_param_dict['shift_fs'] = shift_fs
    signal_param_dict['window_fs'] = window_fs
    signal_param_dict['half_window'] = half_window
    signal_param_dict['start_offset'] = start_offset
    signal_param_dict['end_offset'] = end_offset

    return signal_param_dict


def test_for_bad_window(start, stop, shape, window):
    # if the window_begin is less than 0 or
    # check if onset is within limits
    # if the window_end is less than 0 or
    # if the window_end is outside the signal
    # if there are not enough frames in the window
    return (start < 0 or start > shape[0] or stop < 0 or stop > shape[0]
            or stop - start < window or stop - start < 0)
