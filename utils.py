import math
import os
import random

import numpy as np
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

    Args:
        CONFIG (dict): configuration information
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
    """Parse conversations to return examples

    Args:
        file (str): conversation file
        delim (str): text delimiter in the conversation file
        ex_words (list): words to be excluded
        vocab_str (str, optional): vocabulary to use. (Standard/SentencePiece)
                                   defaults to 'std'.

    Returns:
        list: example tuples
    """
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


def calculate_windows_params(CONFIG, gram, param_dict):
    """Add offset to window begin, end and count bins

    Args:
        CONFIG (dict): configuration information
        gram (tuple): tuple object of word and its corresponding signal window
        param_dict (dict): signal parameters

    Returns:
        int: sequence length
        int: index for window start
        int: index for window end
        int: number of bins in that window
    """
    seq_length = gram[3] - gram[2]
    begin_window = gram[2] + param_dict['start_offset']
    end_window = gram[3] + param_dict['end_offset']

    if CONFIG["classify"] and CONFIG["ngrams"] and not CONFIG["nseq"]:
        bin_size = 50
    elif CONFIG["classify"] and not CONFIG["ngrams"]:  # fixed number of bins
        half_window = param_dict["half_window"]
        bin_size = len(range(-half_window, half_window, param_dict["bin_fs"]))
    elif CONFIG["classify"] and CONFIG["ngrams"] and CONFIG["nseq"]:
        bin_size = int(
            math.ceil((end_window - begin_window) / param_dict['bin_fs']))
    else:
        print("Nothing can be done")

    return seq_length, begin_window, end_window, bin_size


def convert_ms_to_fs(CONFIG, fs=512):
    """Convert seconds to frames

    Args:
        CONFIG (dict): Configuration information
        fs (int, optional): Frames per second. Defaults to 512.

    Returns:
        dict: parameters for extracting neural signals
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


def print_cuda_usage(CONFIG):
    """CUDA usage statistics

    Args:
        CONFIG (dict): configuration information
    """
    print('Memory Usage:')
    for i in range(CONFIG["gpus"]):
        max_alloc = round(torch.cuda.max_memory_allocated(i) / 1024**3, 1)
        cached = round(torch.cuda.memory_cached(i) / 1024**3, 1)
        print(f'GPU: {i} Allocated: {max_alloc}G Cached: {cached}G')
