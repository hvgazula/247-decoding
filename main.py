import math
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from transformers import AdamW

from arg_parser import arg_parser
from build_matrices import build_design_matrices_seq2seq
from config import build_config
from dl_utils import Brain2TextDataset, MyCollator
from filter_utils import filter_by_labels, filter_by_signals
from model_utils import return_model
from plot_utils import figure5
from rw_utils import bigram_counts_to_csv
from utils import fix_random_seed, transform_labels
from vocab_builder import create_vocab

now = datetime.now()
date_str = now.strftime("%A %d/%m/%Y %H:%M:%S")
results_str = now.strftime("%Y-%m-%d-%H:%M")

args = arg_parser()
CONFIG = build_config(args, results_str)
sys.stdout = open(CONFIG["LOG_FILE"], 'w')

print(f'Start Time: {date_str}')
print(f'Setting Random seed: {CONFIG["seed"]}')
fix_random_seed(CONFIG)

# Model objectives
MODEL_OBJ = {
    "ConvNet10": "classifier",
    "PITOM": "classifier",
    "MeNTALmini": "classifier",
    "MeNTAL": "seq2seq"
}

# GPUs
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args.gpus = min(args.gpus, torch.cuda.device_count())

args.model = args.model.split("_")[0]
classify = False if (args.model in MODEL_OBJ
                     and MODEL_OBJ[args.model] == "seq2seq") else True

if classify:
    pass
else:
    print('Pre-filtering Count: ')
    signals, labels = build_design_matrices_seq2seq(CONFIG,
                                                    delimiter=" ",
                                                    aug_shift_ms=[-1000, -500],
                                                    max_num_bins=0)

    print('Plotting Distribution of Signal Lengths')
    seq_lengths = [seq.shape[0] for seq in signals]
    figure5(CONFIG["SAVE_DIR"], seq_lengths, 'all')

    signals, labels = filter_by_signals(signals, labels, 75)
    assert len(signals) == len(labels), "Size Mismatch: Filter 1"
    print(f'Number of Examples (Post Signal Length Cutoff): {len(signals)}')
    signals, labels = filter_by_labels(signals, labels, 30)
    assert len(signals) == len(labels), "Size Mismatch: Filter 2"
    print(f'Number of Examples (Post Class Size Cutoff): {len(signals)}')

    bigram_counts_to_csv(CONFIG, labels, 'mixed')

    X_train, X_test, y_train, y_test = train_test_split(signals,
                                                        labels,
                                                        stratify=labels,
                                                        test_size=0.30,
                                                        random_state=args.seed)

    bigram_counts_to_csv(CONFIG, y_train, 'train')
    bigram_counts_to_csv(CONFIG, y_test, 'test')

    print(f'Size of Training Set is: {len(X_train)}')
    print(f'Size of Test Set is: {len(X_test)}')

    print('Building Vocabulary')
    word2freq, word_list, n_classes, vocab, i2w = create_vocab(CONFIG,
                                                               y_train,
                                                               classify=False)

    print('Transforming Labels')
    y_train = transform_labels(CONFIG, vocab, y_train)
    y_test = transform_labels(CONFIG, vocab, y_test)

    print('Creating Dataset Objects')
    train_ds = Brain2TextDataset(X_train, y_train)
    valid_ds = Brain2TextDataset(X_test, y_test)

    print('Creating DataLoader Objects')
    my_collator = MyCollator(CONFIG, vocab)
    train_dl = data.DataLoader(train_ds,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=CONFIG["num_cpus"],
                               collate_fn=my_collator)
    valid_dl = data.DataLoader(valid_ds,
                               batch_size=args.batch_size,
                               num_workers=CONFIG["num_cpus"],
                               collate_fn=my_collator)

    model = return_model(args, CONFIG, vocab)

    # Initialize loss and optimizer
    criterion = nn.CrossEntropyLoss()
    step_size = int(math.ceil(len(train_ds) / args.batch_size))
    optimizer = AdamW(model.parameters(),
                      lr=args.lr,
                      weight_decay=args.weight_decay)
    scheduler = None

    # Move model and loss to GPUs
    if args.gpus:
        if args.gpus > 1:
            model = nn.DataParallel(model)

    model.to(DEVICE)

    print('Printing Model Summary')
    with open(os.path.join(CONFIG["SAVE_DIR"], 'model_summary'),
              'w') as file_h:
        print(model, file=file_h)
