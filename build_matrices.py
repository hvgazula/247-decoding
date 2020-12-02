import glob

import numpy as np

from electrode_utils import return_electrode_array
from gram_utils import generate_bigrams, generate_unigrams, remove_duplicates
from utils import (calculate_windows_params, convert_ms_to_fs,
                   return_conversations, return_examples, test_for_bad_window)


def build_design_matrices(CONFIG,
                          fs=512,
                          delimiter=',',
                          aug_shift_ms=[-500, -250, 250]):
    """Build examples and labels for the model

    Args:
        CONFIG (dict): configuration information
        fs (int, optional): frames per second. Defaults to 512.
        delimiter (str, optional): conversation delimier. Defaults to ','.
        aug_shift_ms (list, optional): shifts for data augmentation.
        Defaults to [-500, -250, 250].

    Returns:
        tuple: (signals, labels)

    Misc:
        signals: neural activity data
        labels: words/n-grams/sentences
    """
    exclude_words = CONFIG["exclude_words"]
    signal_param_dict = convert_ms_to_fs(CONFIG)

    convs = return_conversations(CONFIG)
    cumsum_electrodes = list(np.cumsum(CONFIG['max_electrodes']))
    cumsum_electrodes.insert(0, 0)

    signals, labels = [], []

    full_signal, binned_signal = [], []
    full_stitch_index, bin_stitch_index = [], []
    for conversation, suffix, idx, electrodes in convs:

        try:  # Check if files exists
            datum_fn = glob.glob(conversation + suffix)[0]
        except IndexError:
            print('File DNE: ', conversation + suffix)
            continue

        # Extract electrode data
        ecogs = return_electrode_array(conversation, electrodes)
        if not ecogs.size:
            print(f'Bad Conversation: {conversation}')
            continue

        if CONFIG['pickle']:
            full_signal.append(ecogs)
            full_stitch_index.append(ecogs.shape[0])

            split_indices = np.arange(32, ecogs.shape[0], 32)
            convo_binned_signal = np.vsplit(ecogs, split_indices)
            if convo_binned_signal[-1].shape[0] < 32:
                convo_binned_signal.pop(-1)

            mean_binned_signal = [
                np.mean(split, axis=0) for split in convo_binned_signal
            ]

            mean_binned_signal = np.vstack(mean_binned_signal)
            bin_stitch_index.append(mean_binned_signal.shape[0])

            binned_signal.append(mean_binned_signal)

            continue

        examples = return_examples(datum_fn, delimiter, exclude_words,
                                   CONFIG["vocabulary"])

        if CONFIG["classify"] and not CONFIG["ngrams"]:
            unigrams = generate_unigrams(examples)
            if not unigrams:
                print(f'Bad Conversation: {conversation}')
                continue
            grams = set(unigrams)  # Removing duplicates
        else:
            bigrams = generate_bigrams(examples)
            if not bigrams:
                print(f'Bad Conversation: {conversation}')
                continue
            grams = remove_duplicates(bigrams)

        for gram in grams:
            (seq_length, start_onset, end_onset,
             n_bins) = (calculate_windows_params(CONFIG, gram,
                                                 signal_param_dict))

            if (seq_length <= 0):
                continue

            if test_for_bad_window(start_onset, end_onset, ecogs.shape):
                continue

            labels.append(gram[0])
            word_signal = np.zeros((n_bins, CONFIG['num_features']),
                                   np.float32)

            for i, f in enumerate(
                    np.array_split(ecogs[start_onset:end_onset, :],
                                   n_bins,
                                   axis=0)):
                word_signal[
                    i, cumsum_electrodes[idx]:cumsum_electrodes[idx +
                                                                1]] = f.mean(
                                                                    axis=0)

            # TODO Data Augmentation
            signals.append(word_signal)

    if CONFIG['pickle']:
        full_signal = np.concatenate(full_signal)
        full_stitch_index = np.cumsum(full_stitch_index)

        binned_signal = np.vstack(binned_signal)
        bin_stitch_index = np.cumsum(bin_stitch_index)
        return full_signal, full_stitch_index, binned_signal, bin_stitch_index

    print(f'Total number of conversations: {len(convs)}')
    print(f'Number of samples is: {len(signals)}')
    print(f'Number of labels is : {len(labels)}')

    print(f'Maximum Sequence Length: {max([len(i) for i in signals])}')

    assert len(labels) == len(signals), "Bad Shape for Lengths"

    return signals, labels
