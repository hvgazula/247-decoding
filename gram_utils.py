import pandas as pd


def transform_labels(CONFIG, vocabulary, label_list):
    """Convert labels to indices

    Args:
        CONFIG (dict): configuration information
        vocabulary (dict): self-explanatory
        label_list (list): labels to be transformed

    Returns:
        list (of lists):

    Example:
        >>> [['hi', 'hello']]
                --> [['<s>', 'hi' ,'hello', '</s>']]
                    --> [[0, 23, 19, 1]]
    """
    classify = CONFIG["classify"]

    if classify:
        transformed_label_list = [vocabulary[x] for x in label_list]
    else:  # add start and end tokens
        start_tok = CONFIG["begin_token"]
        stop_tok = CONFIG["end_token"]

        transformed_label_list = []
        for word_pair in label_list:
            word_pair = [vocabulary[x] for x in word_pair]
            word_pair.insert(0, vocabulary[start_tok])  # Add start token
            word_pair.append(vocabulary[stop_tok])  # Add end token
            transformed_label_list.append(word_pair)

    return transformed_label_list


def generate_unigrams(examples):
    """Generate examples for classification

    Args:
        examples (tuple): tuples parsed from conversation

    Returns:
        list: words/labels
    """
    my_grams = []
    for example in examples:
        if len(example[0]) > 1:
            continue
        if not example[1]:
            continue
        my_grams.append((example[0][0], *example[1:]))
    return my_grams


def remove_duplicates(grams):
    """Remove duplicate n-grams

    Args:
        grams (tuple): n-gram tuples from conversations

    Returns:
        list: of examples (tuple/list)
    
    TODO: needs revamp
    """
    df = pd.DataFrame(grams)
    df[['fw', 'sw']] = pd.DataFrame(df[0].tolist())
    df = df.drop(columns=[0]).drop_duplicates()
    df[0] = df[['fw', 'sw']].values.tolist()
    df = df.drop(columns=['fw', 'sw'])
    df = df[sorted(df.columns)]
    return list(df.to_records(index=False))


def generate_bigrams(examples):
    '''if the first set already has two words and is speaker 1
        if the second set already has two words and is speaker 1
        the onset of the first word is earlier than the second word
    '''
    my_grams = []
    for first, second in zip(examples, examples[1:]):
        len1, len2 = len(first[0]), len(second[0])
        if first[1] and len1 == 2:
            my_grams.append(first)
        if second[1] and len2 == 2:
            my_grams.append(second)
        if ((first[1] and second[1]) and (len1 == 1 and len2 == 1)
                and (first[2] < second[2])):
            ak = (first[0] + second[0], True, first[2], second[2])
            my_grams.append(ak)
    return my_grams
