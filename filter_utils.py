from collections import Counter
from itertools import compress


def filter_by_labels(CONFIG, signals, labels, set_min_samples):
    """Return filtered labels and corresponding features

    Args:
        CONFIG (dict): configuration information
        signals (iterable): neural data
        labels (iterable): targets words
        set_min_samples (int): minimum number of samples

    Returns:
        tuple: (filtered_examples, filtered_labels)
    """    
    classify = CONFIG["classify"]
    label_flag = return_label_flag(labels, set_min_samples, classify=classify)
    signals = list(compress(signals, label_flag))
    labels = list(compress(labels, label_flag))
    return signals, labels


def filter_by_signals(signals, labels, set_max_seq_length):
    """Return filtered signals and corresponding labels

    Args:
        CONFIG (dict): configuration information
        signals (iterable): neural data
        labels (iterable): targets words
        set_max_seq_length (int): desired sequence length (upper limit)

    Returns:
        tuple: (filtered_examples, filtered_labels)
    """
    signal_flag = return_signal_flag(signals, set_max_seq_length)
    signals = list(compress(signals, signal_flag))
    labels = list(compress(labels, signal_flag))
    return signals, labels


def return_label_flag(labels, set_min_samples, classify=True):
    """Check for minimum number of samples in each class

    Args:
        labels (iterable): list of labels/targets
        set_min_samples (int): minimum clas size
        classify (bool, optional): flag indication classification/seq2seq.
                                   Defaults to True.

    Returns:
        list: filtered labels
    """
    label_counter = label_counts(labels, classify=classify)
    select_labels = [
        label if classify else list(label)
        for label, count in label_counter.items() if count >= set_min_samples
    ]
    label_flag = [label in select_labels for label in labels]
    return label_flag


def return_signal_flag(signals, set_max_seq_length):
    """

    Args:
        signals (iterable): signals/features
        set_max_seq_length (int): desired maximum sequence length

    Returns:
        bool: boolean vector indicating if seq_length < max_seq_length
    """
    seq_lengths = [len(signal) for signal in signals]
    signal_flag = [
        seq_length <= set_max_seq_length for seq_length in seq_lengths
    ]
    return signal_flag


def label_counts(labels, classify=True):
    """Return counts of labels

    Args:
        labels (list): labels -> n-grams/bigrams/sentences
        classify (bool, optional): Flag describing classification/seq2seq.
                                   Defaults to True.

    Returns:
        counter: counter object with counts of each unique label
    """
    if classify:
        return Counter(labels)
    else:
        return Counter([tuple(label) for label in labels])
