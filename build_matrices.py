import glob

import numpy as np

from electrode_utils import return_electrode_array
from gram_utils import generate_bigrams, generate_unigrams, remove_duplicates
from utils import (calculate_windows_params, convert_ms_to_fs,
                   return_conversations, return_examples, test_for_bad_window)


def build_design_matrices_seq2seq(CONFIG,
                                  fs=512,
                                  delimiter=',',
                                  aug_shift_ms=[-500, -250, 250],
                                  max_num_bins=None,
                                  remove_unks=True):
    """[summary]

    Args:
        CONFIG ([type]): [description]
        fs (int, optional): [description]. Defaults to 512.
        delimiter (str, optional): [description]. Defaults to ','.
        aug_shift_ms (list, optional): [description].
        Defaults to [-500, -250, 250].
        max_num_bins ([type], optional): [description]. Defaults to None.
        remove_unks (bool, optional): [description]. Defaults to True.

    Returns:
        [type]: [description]
    """
    exclude_words = CONFIG["exclude_words"]
    signal_param_dict = convert_ms_to_fs(CONFIG)

    convs = return_conversations(CONFIG)
    cumsum_electrodes = list(np.cumsum(CONFIG['max_electrodes']))
    cumsum_electrodes.insert(0, 0)

    signals, labels, seq_lengths = [], [], []
    for conversation, suffix, idx, electrodes in convs[:15]:
        # Check if files exists, if it doesn't go to next
        try:
            datum_fn = glob.glob(conversation + suffix)[0]
        except IndexError:
            continue

        if not datum_fn:
            print('File DNE: ', conversation + suffix)
            continue

        # Extract electrode data
        ecogs = return_electrode_array(conversation, electrodes)
        if not ecogs.size:
            print(f'Bad Conversation: {conversation}')
            continue

        examples = return_examples(datum_fn, delimiter, exclude_words,
                                   CONFIG["vocabulary"])
        bigrams = generate_bigrams(examples)

        if not bigrams:
            print(f'Bad Conversation: {conversation}')
            continue
        bigrams = remove_duplicates(bigrams)

        for bigram in bigrams:
            (seq_length, start_onset, end_onset,
             n_bins) = (calculate_windows_params(CONFIG, bigram,
                                                 signal_param_dict))

            if (seq_length <= 0) or (max_num_bins and n_bins > max_num_bins):
                continue

            seq_lengths.append(seq_length)

            if test_for_bad_window(start_onset, end_onset, ecogs.shape,
                                   signal_param_dict['window_fs']):
                continue

            labels.append(bigram[0])
            word_signal = np.zeros((n_bins, sum(CONFIG['max_electrodes'])),
                                   np.float32)
            for i, f in enumerate(
                    np.array_split(ecogs[start_onset:end_onset, :],
                                   n_bins,
                                   axis=0)):
                word_signal[i, cumsum_electrodes[idx]:cumsum_electrodes[
                    idx + 1]] = f.mean(axis=0)

            # TODO: Data Augmentation
            signals.append(word_signal)

    print(f'Maximum Sequence Length (Preset): {max_num_bins}')

    print(f'Total number of conversations: {len(convs)}')
    print(f'Number of samples is: {len(signals)}')
    print(f'Number of labels is : {len(labels)}')

    print(f'Maximum Sequence Length: {max([len(i) for i in signals])}')

    assert len(labels) == len(signals), "Bad Shape for Lengths"

    return signals, labels


def build_design_matrices_classification(CONFIG,
                                         fs=512,
                                         delimiter=',',
                                         aug_shift_ms=[-500, -250, 250]):

    exclude_words = CONFIG["exclude_words"]
    signal_param_dict = convert_ms_to_fs(CONFIG)

    half_window = signal_param_dict["half_window"]
    n_bins = len(range(-half_window, half_window, signal_param_dict["bin_fs"]))

    convs = return_conversations(CONFIG)
    cumsum_electrodes = list(np.cumsum(CONFIG['max_electrodes']))
    cumsum_electrodes.insert(0, 0)

    signals, labels = [], []
    for conversation, suffix, idx, electrodes in convs[:15]:
        # Check if files exists, if it doesn't go to next
        try:
            datum_fn = glob.glob(conversation + suffix)[0]
        except IndexError:
            print('File DNE: ', conversation + suffix)
            continue

        # Extract electrode data
        ecogs = return_electrode_array(conversation, electrodes)
        if not ecogs.size:
            print(f'Skipping bad conversation: {conversation}')
            continue

        examples = return_examples(datum_fn, delimiter, exclude_words,
                                   CONFIG["vocabulary"])

        examples = generate_unigrams(
            examples)  # generating unigrams from conversations
        unigrams = set(examples)  # Removing duplicates

        for unigram in unigrams:
            (seq_length, start_onset, end_onset,
             n_bins) = (calculate_windows_params(CONFIG, unigram,
                                                 signal_param_dict))

            if (seq_length <= 0):
                continue

            if test_for_bad_window(start_onset, end_onset, ecogs.shape,
                                   signal_param_dict['window_fs']):
                continue

            labels.append(unigram[0])
            word_signal = np.zeros((n_bins, sum(CONFIG['max_electrodes'])),
                                   np.float32)
            for i, f in enumerate(
                    np.array_split(ecogs[start_onset:end_onset, :],
                                   n_bins,
                                   axis=0)):
                word_signal[i, cumsum_electrodes[idx]:cumsum_electrodes[
                    idx + 1]] = f.mean(axis=0)

            # TODO: Data Augmentation
            signals.append(word_signal)

    print(f'Total number of conversations: {len(convs)}')
    print(f'Number of samples is: {len(signals)}')
    print(f'Number of labels is: {len(labels)}')

    print(f'Maximum Sequence Length: {max([len(i) for i in signals])}')

    assert len(labels) == len(signals), "Bad Shape for Lengths"

    return signals, labels
