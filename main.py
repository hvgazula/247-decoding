import itertools
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from transformers import AdamW

from arg_parser import arg_parser
from build_matrices import build_design_matrices
from config import build_config
from dl_utils import Brain2TextDataset, MyCollator
from eval_utils import evaluate_roc
from filter_utils import filter_by_labels, filter_by_signals
from gram_utils import transform_labels
from model_utils import return_model
from plot_utils import figure5, plot_training
from rw_utils import bigram_counts_to_csv, print_model
from seq_eval_utils import (bigram_freq_excel, create_excel_preds,
                            translate_neural_signal)
from train_eval import train, valid
from utils import fix_random_seed, print_cuda_usage
from vocab_builder import create_vocab

now = datetime.now()
date_str = now.strftime("%A %d/%m/%Y %H:%M:%S")
results_str = now.strftime("%Y-%m-%d-%H:%M")

args = arg_parser()
CONFIG = build_config(args, results_str)
# sys.stdout = open(CONFIG["LOG_FILE"], 'w')

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE, type(DEVICE))
print(f'Start Time: {date_str}')
print(f'Setting Random seed: {CONFIG["seed"]}')
fix_random_seed(CONFIG)

classify = CONFIG['classify']

print('Building Design Matrix...')
signals, labels = build_design_matrices(CONFIG,
                                        delimiter=" ",
                                        aug_shift_ms=[-1000, -500])

print('Plotting Distribution of Signal Lengths')
seq_lengths = [len(signal) for signal in signals]
figure5(CONFIG["SAVE_DIR"], seq_lengths, 'all')

signals, labels = filter_by_signals(signals, labels, 75)
assert len(signals) == len(labels), "Size Mismatch: Filter 1"
print(f'Number of Examples (Post Signal Length Cutoff): {len(signals)}')

signals, labels = filter_by_labels(CONFIG, signals, labels, 20)
assert len(signals) == len(labels), "Size Mismatch: Filter 2"
print(f'Number of Examples (Post Class Size Cutoff): {len(signals)}')

bigram_counts_to_csv(CONFIG, labels, data_str='mixed')

X_train, X_test, y_train, y_test = train_test_split(
    signals,
    labels,
    stratify=labels,
    test_size=0.30,
    random_state=CONFIG["seed"])

bigram_counts_to_csv(CONFIG, y_train, data_str='train')
bigram_counts_to_csv(CONFIG, y_test, data_str='test')

print(f'Size of Training Set is: {len(X_train)}')
print(f'Size of Test Set is: {len(X_test)}')

print('Building Vocabulary')
word2freq, word_list, n_classes, vocab, i2w = create_vocab(CONFIG, y_train)

print('Transforming Labels')
y_train = transform_labels(CONFIG, vocab, y_train)
y_test = transform_labels(CONFIG, vocab, y_test)

print('Creating Dataset Objects')
train_ds = Brain2TextDataset(X_train, y_train)
valid_ds = Brain2TextDataset(X_test, y_test)

print('Creating DataLoader Objects')
my_collator = None if classify else MyCollator(CONFIG, vocab)

train_dl = data.DataLoader(train_ds,
                           batch_size=CONFIG["batch_size"],
                           shuffle=True,
                           num_workers=CONFIG["num_cpus"],
                           collate_fn=my_collator)
valid_dl = data.DataLoader(valid_ds,
                           batch_size=CONFIG["batch_size"],
                           num_workers=CONFIG["num_cpus"],
                           collate_fn=my_collator)

model = return_model(CONFIG, vocab)

# Initialize loss and optimizer
criterion = nn.CrossEntropyLoss()
step_size = int(math.ceil(len(train_ds) / CONFIG["batch_size"]))
optimizer = AdamW(model.parameters(),
                  lr=CONFIG["lr"],
                  weight_decay=CONFIG["weight_decay"])
scheduler = None

# Move model and loss to GPUs
if CONFIG["gpus"]:
    if CONFIG["gpus"] > 1:
        model = nn.DataParallel(model)

model.to(DEVICE)
print_model(CONFIG, model)

print("\nTraining on %d GPU(s) with batch_size %d for %d epochs" %
      (CONFIG["gpus"], CONFIG["batch_size"], CONFIG["epochs"]))
sys.stdout.flush()

best_val_loss = float("inf")
best_model = model
history = {
    'train_loss': [],
    'train_acc': [],
    'valid_loss': [],
    'valid_acc': []
}

epoch = 0
model_name = "%s%s.pt" % (CONFIG["SAVE_DIR"], CONFIG["model"])

lr = CONFIG["lr"]
for epoch in range(1, CONFIG["epochs"] + 1):
    epoch_start_time = time.time()
    print(f'Epoch: {epoch:02}')
    print('\tTrain: ', end='')
    train_loss, train_acc = train(
        train_dl,
        model,
        criterion,
        list(range(CONFIG["gpus"])),
        DEVICE,
        optimizer,
        scheduler=scheduler,
        seq2seq=not classify,
        pad_idx=vocab[CONFIG["pad_token"]] if not classify else -1)
    for param_group in optimizer.param_groups:
        if 'lr' in param_group:
            print(' | lr {:1.2E}'.format(param_group['lr']))
            break
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    print('\tValid: ', end='')
    with torch.no_grad():
        valid_loss, valid_acc = valid(
            valid_dl,
            model,
            criterion,
            DEVICE,
            temperature=CONFIG["temp"],
            seq2seq=not classify,
            pad_idx=vocab[CONFIG["pad_token"]] if not classify else -1)
    history['valid_loss'].append(valid_loss)
    history['valid_acc'].append(valid_acc)

    # Store best model so far
    if valid_loss < best_val_loss:
        best_model, best_val_loss = model, valid_loss
        model_to_save = best_model.module if hasattr(best_model,
                                                     'module') else best_model
        torch.save(model_to_save, model_name)

    # Additional Info when using cuda
    print_cuda_usage(CONFIG) if DEVICE.type == 'cuda' else None

print('Printing Loss Curves')
plot_training(history,
              CONFIG["SAVE_DIR"],
              title="%s_lr%s" % (CONFIG["model"], CONFIG["lr"]))

print("Evaluating predictions on test set")
# Load best model
best_model = torch.load(model_name)

if classify:
    print('Need to work on this part of the code')
else:
    vocab_len = len(vocab)

    (valid_all_trg_y, valid_topk_preds, valid_topk_preds_scores,
     valid_all_preds) = translate_neural_signal(CONFIG, vocab, DEVICE,
                                                best_model, valid_dl,
                                                vocab_len)
    (train_all_trg_y, train_topk_preds, train_topk_preds_scores,
     train_all_preds) = translate_neural_signal(CONFIG, vocab, DEVICE,
                                                best_model, train_dl,
                                                vocab_len)

    valid_preds_df = create_excel_preds(valid_all_trg_y, valid_topk_preds, i2w)
    train_preds_df = create_excel_preds(train_all_trg_y, train_topk_preds, i2w)

    valid_preds_df.to_excel(os.path.join(CONFIG["SAVE_DIR"], 'valid_set.xlsx'),
                            index=False)
    train_preds_df.to_excel(os.path.join(CONFIG["SAVE_DIR"], 'train_set.xlsx'),
                            index=False)

    train_freqs = {vocab[key]: val for key, val in word2freq.items()}

    do_plot = True
    remove_tokens = [
        CONFIG["begin_token"], CONFIG["end_token"], CONFIG["oov_token"],
        CONFIG["pad_token"]
    ]
    word_end_idex = []

    # doing it for word1
    true = np.array(valid_preds_df['word1'].replace(vocab).tolist())
    labels = np.zeros((true.size, true.max() + 1))
    labels[np.arange(true.size), true] = 1
    predictions = valid_all_preds.numpy()[:, n_classes * 0:n_classes * 1]
    evaluate_roc(predictions,
                 labels,
                 i2w,
                 train_freqs,
                 CONFIG["SAVE_DIR"],
                 do_plot,
                 given_thresholds=None,
                 title='word1',
                 suffix='word1',
                 min_train=10,
                 tokens_to_remove=remove_tokens)

    # doing it for word2
    true = np.array(valid_preds_df['word2'].replace(vocab).tolist())
    labels = np.zeros((true.size, true.max() + 1))
    labels[np.arange(true.size), true] = 1
    predictions = valid_all_preds.numpy()[:, n_classes * 1:n_classes * 2]
    evaluate_roc(predictions,
                 labels,
                 i2w,
                 train_freqs,
                 CONFIG["SAVE_DIR"],
                 do_plot,
                 given_thresholds=None,
                 title='word2',
                 suffix='word2',
                 min_train=10,
                 tokens_to_remove=remove_tokens)

    # Post-processing for bigrams as classes
    # EXAMPLE
    # a = np.array([[1,2],[3,4]])
    # np.repeat(a, 2, axis=2)
    # np.tile(a, (1, 2))
    softmax = nn.Softmax(dim=1)

    valid_all_preds_first = valid_all_preds[:, n_classes * 0:n_classes * 1]
    valid_all_preds_second = valid_all_preds[:, n_classes * 1:n_classes * 2]

    valid_all_preds_first_repeat = torch.repeat_interleave(
        valid_all_preds_first, n_classes,
        dim=1)  # this is similar to np.repeat
    valid_all_preds_second_repeat = valid_all_preds_second.repeat(
        (1, n_classes))  # this is similar to np.tile

    valid_all_preds_bigram = valid_all_preds_first_repeat * \
        valid_all_preds_second_repeat  # hadamard product
    valid_all_preds_bigram = softmax(
        valid_all_preds_bigram)  # softmax in dim=1

    raw_train_df = bigram_freq_excel(CONFIG, y_train, word2freq, i2w,
                                     "625_bi-gram-freq-train.xlsx")
    raw_valid_df = bigram_freq_excel(CONFIG,
                                     y_test,
                                     word2freq,
                                     i2w,
                                     "625_bi-gram-freq-valid.xlsx",
                                     ref_data=raw_train_df)

    abc = [p for p in itertools.product(vocab.keys(), repeat=2)]
    bigram_i2w = {i: '_'.join(words) for i, words in enumerate(abc)}
    new_df = pd.DataFrame(abc, columns=['word1', 'word2'])
    new_df['bigram_index'] = new_df.index
    new_valid_preds_df = pd.merge(valid_preds_df,
                                  new_df,
                                  how='left',
                                  on=['word1', 'word2'])

    # doing it for bigrams
    true = np.array(new_valid_preds_df['bigram_index'])
    labels = np.zeros((true.size, n_classes**2))
    labels[np.arange(true.size), true] = 1
    predictions = valid_all_preds_bigram

    raw_train_df = bigram_freq_excel(CONFIG, y_train, word2freq, i2w,
                                     "676_bi-gram-freq-train.xlsx")
    raw_valid_df = bigram_freq_excel(CONFIG,
                                     y_test,
                                     word2freq,
                                     i2w,
                                     "676_bi-gram-freq-valid.xlsx",
                                     ref_data=raw_train_df)
    new_raw_train_df = pd.merge(raw_train_df,
                                new_df,
                                how='left',
                                on=['word1', 'word2'])

    a = new_raw_train_df[['bigram_index', 'Count']]
    kabc = a.to_dict('records')
    bigram_train_freqs = {item['bigram_index']: item['Count'] for item in kabc}

    evaluate_roc(predictions,
                 labels,
                 bigram_i2w,
                 bigram_train_freqs,
                 CONFIG["SAVE_DIR"],
                 do_plot,
                 given_thresholds=None,
                 title='bigram',
                 suffix='bigram',
                 min_train=5,
                 tokens_to_remove=remove_tokens)

    print("so far so good")
