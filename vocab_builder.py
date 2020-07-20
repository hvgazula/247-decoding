from collections import Counter, OrderedDict
from itertools import chain

from rw_utils import save_word_counter


def create_vocab(CONFIG, y_train):
    """Create vocabulary for the model

    Args:
        CONFIG (dict): configuration information
        y_train (list): label data

    Returns:
        dict: frequency of words
        int: length of vocabulary
        dict: word to index dictionary
        dict: index to word dictionary

    Example:
        >>> a = [['w11', 'w21'], ['w12', 'w22'], ['w13', 'w23']]
        >>> print(chain.from_iterable(a))
        >>> ['w11', 'w21', 'w12', 'w22', 'w11', 'w23']
    """
    classify = CONFIG["classify"]

    word_freq = Counter()
    min_freq = CONFIG["vocab_min_freq"]
    
    word_freq.update(y_train if classify else chain.from_iterable(y_train))

    tokens_to_add = [
        CONFIG["begin_token"], CONFIG["end_token"], CONFIG["oov_token"],
        CONFIG["pad_token"]
    ]

    start_index = 0 if classify else len(tokens_to_add)

    vocab = sorted(word_freq.keys())
    w2i = {word: i for i, word in enumerate(vocab, start_index)}

    if not classify:
        w2f_tok, w2i_tok = {}, {}
        for i, token in enumerate(tokens_to_add):
            w2f_tok[token] = -1
            w2i_tok[token] = i
        word_freq.update(w2f_tok)
        w2i.update(w2i_tok)
        vocab.extend(tokens_to_add)
        w2i = OrderedDict(sorted(w2i.items(), key=lambda kv: kv[1]))

    n_classes = len(vocab)
    i2w = {i: word for word, i in w2i.items()}

    print("Vocabulary size (min_freq=%d): %d" % (min_freq, len(word_freq)))
    print(f'Total number of classes in the dataset is: {n_classes}')

    save_word_counter(CONFIG, word_freq)

    return word_freq, n_classes, w2i, i2w
