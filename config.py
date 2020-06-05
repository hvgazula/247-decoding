import os


def return_config_dict():
    '''
    exclude_words_class: words to be excluded from the classifier vocabulary
    exclude_words: words to be excluded from the tranformer vocabulary
    log_interval:
    '''
    CONFIG = {
        "begin_token":
        "<s>",
        "datum_suffix": ("conversation_trimmed", "trimmed"),
        "end_token":
        "</s>",
        "exclude_words_class": [
            "sp", "{lg}", "{ns}", "it", "a", "an", "and", "are", "as", "at",
            "be", "being", "by", "for", "from", "is", "of", "on", "that",
            "that's", "the", "there", "there's", "this", "to", "their", "them",
            "these", "he", "him", "his", "had", "have", "was", "were", "would"
        ],
        "exclude_words": ["sp", "{lg}", "{ns}"],
        "log_interval":
        32,
        "data_dir":
        "/scratch/gpfs/hgazula/brain2en-seq2seq-data",
        "num_cpus":
        8,
        "oov_token":
        "<unk>",
        "pad_token":
        "<pad>",
        "print_pad":
        120,
        "train_convs":
        '-train-convs.txt',
        "valid_convs":
        '-valid-convs.txt',
        "vocabulary":
        'std'
    }

    return CONFIG


def build_config(args, results_str):

    CONFIG = return_config_dict()

    # Format directory logistics
    CONV_DIRS = [
        CONFIG["data_dir"] + '/%s-conversations/' % i for i in args.subjects
    ]
    META_DIRS = [
        CONFIG["data_dir"] + '/%s-metadata/' % i for i in args.subjects
    ]
    SAVE_DIR = './Results/%s-%s-%s/' % (results_str, '+'.join(
        args.subjects), args.model)
    LOG_FILE = SAVE_DIR + 'output'
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    DIR_DICT = dict(CONV_DIRS=CONV_DIRS,
                    META_DIRS=META_DIRS,
                    SAVE_DIR=SAVE_DIR,
                    LOG_FILE=LOG_FILE)

    CONFIG.update(DIR_DICT)

    if len(args.subjects) == 1:
        if args.subjects[0] == '625':
            CONFIG["datum_suffix"] = [CONFIG["datum_suffix"][0]]
        elif args.subjects[0] == '676':
            CONFIG["datum_suffix"] = [CONFIG["datum_suffix"][1]]

    CONFIG.update(vars(args))

    CONFIG["electrode_list"] = [
        range(1, k + 1) for k in CONFIG["max_electrodes"]
    ]

    return CONFIG
