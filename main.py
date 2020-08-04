import math
import sys
import time
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from transformers import AdamW

from arg_parser import arg_parser
from build_matrices import build_design_matrices
from config import build_config
from dl_utils import Brain2TextDataset, MyCollator, pitom_collate
from eval_utils import evaluate_roc, evaluate_topk
from filter_utils import filter_by_labels, filter_by_signals
from gram_utils import transform_labels
from model_utils import return_model
from plot_utils import figure5, plot_training
from rw_utils import print_model, tabulate_and_print
from s2s_eval_utils import (bigram_accuracy_report, calc_bigram_train_freqs,
                            concatenate_bigrams, count_unique_predictions,
                            create_excel_preds, return_bigram_proba,
                            return_bigram_vocab, save_bigram_counts,
                            topk_accuracy_report, word_wise_roc)
from s2s_inference import classify_neural_signal, translate_neural_signal
from train_eval import train, valid
from utils import fix_random_seed, print_cuda_usage
from vocab_builder import create_vocab

start_time = datetime.now()
date_str = start_time.strftime("%A %m/%d/%Y %H:%M:%S")
results_str = start_time.strftime("%Y%m%d%H%M")  # results folder prefix

args = arg_parser()  # parse command line arguments
CONFIG = build_config(args, results_str)
# sys.stdout = open(CONFIG["LOG_FILE"], 'w')

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Start Time: {date_str}')
print(f'Setting Random seed: {CONFIG["seed"]}')
fix_random_seed(CONFIG)

classify = CONFIG['classify']  # if classification/seq2seq

print('Building Design Matrix...')
signals, labels = build_design_matrices(CONFIG,
                                        delimiter=" ",
                                        aug_shift_ms=[-1000, -500])

print('Plotting Distribution of Signal Lengths')
seq_lengths = [len(signal) for signal in signals]
figure5(CONFIG["SAVE_DIR"], seq_lengths, 'all')

if CONFIG["classify"] and CONFIG["ngrams"]:
    labels = ['_'.join(label) for label in labels]

signals, labels = filter_by_signals(signals, labels, CONFIG["seq_len_limit"])
assert len(signals) == len(labels), "Size Mismatch: Filter 1"
print(f'Number of Examples (Post Signal Length Cutoff): {len(signals)}')

signals, labels = filter_by_labels(CONFIG, signals, labels,
                                   CONFIG["vocab_min_freq"])
assert len(signals) == len(labels), "Size Mismatch: Filter 2"
print(f'Number of Examples (Post Class Size Cutoff): {len(signals)}')

X_train, X_test, y_train, y_test = train_test_split(
    signals,
    labels,
    stratify=labels,
    test_size=0.30,
    random_state=CONFIG["seed"])

print(f'Size of Training Set is: {len(X_train)}')
print(f'Size of Test Set is: {len(X_test)}')

print('Building Vocabulary')
word2freq, n_classes, vocab, i2w = create_vocab(CONFIG, y_train)

print('Transforming Labels')
y_train = transform_labels(CONFIG, vocab, y_train)
y_test = transform_labels(CONFIG, vocab, y_test)

print('Creating Dataset Objects')
train_ds = Brain2TextDataset(X_train, y_train)
valid_ds = Brain2TextDataset(X_test, y_test)

print('Creating DataLoader Objects')
if classify and not CONFIG["nseq"]:
    my_collator = None
elif classify and CONFIG["nseq"]:
    my_collator = pitom_collate
else:
    my_collator = MyCollator(CONFIG, vocab)

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

best_val_loss = float("inf")
best_model = model
history = {
    'train_loss': [],
    'train_acc': [],
    'valid_loss': [],
    'valid_acc': []
}

model_name = "%s%s.pt" % (CONFIG["SAVE_DIR"], CONFIG["model"])

lr = CONFIG["lr"]
for epoch in range(1, CONFIG["epochs"] + 1):
    print(f'Epoch: {epoch:02}')
    epoch_start_time = time.time()

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

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    print('\n\tValid: ', end='')
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
plot_training(CONFIG, history)

print("Evaluating predictions on test set")
best_model = torch.load(model_name)  # Load best model

if classify:
    all_trg_y, topk_preds, topk_preds_scores, all_preds = classify_neural_signal(CONFIG, vocab, DEVICE, best_model,
                                            valid_dl)

    # Make categorical
    n_examples = len(y_test)
    categorical = np.zeros((n_examples, n_classes), dtype=np.float32)
    categorical[np.arange(n_examples), y_test] = 1

    train_freq = Counter(y_train)

    print("Evaluating top-k")
    evaluate_topk(all_preds,
                  np.array(y_test),
                  i2w,
                  train_freq,
                  CONFIG["SAVE_DIR"],
                  suffix='val',
                  min_train=args.vocab_min_freq)

    print("Evaluating ROC-AUC")
    evaluate_roc(all_preds,
                 categorical,
                 i2w,
                 train_freq,
                 CONFIG["SAVE_DIR"],
                 do_plot=not args.no_plot,
                 min_train=args.vocab_min_freq)

    if CONFIG["ngrams"]:
        valid_preds_df = create_excel_preds(all_trg_y, topk_preds, all_preds, i2w)

else:
    print("Start of postprocessing seq2seq results")
    (train_all_trg_y, train_topk_preds, train_topk_preds_scores,
     train_all_preds) = translate_neural_signal(CONFIG, vocab, DEVICE,
                                                best_model, train_dl)
    (valid_all_trg_y, valid_topk_preds, valid_topk_preds_scores,
     valid_all_preds) = translate_neural_signal(CONFIG, vocab, DEVICE,
                                                best_model, valid_dl)

    valid_preds_df = create_excel_preds(valid_all_trg_y, valid_topk_preds,
                                        valid_all_preds, i2w)
    train_preds_df = create_excel_preds(train_all_trg_y, train_topk_preds,
                                        train_all_preds, i2w)

    count_unique_predictions(CONFIG, valid_preds_df, 'word1')
    count_unique_predictions(CONFIG, valid_preds_df, 'word2')

    tabulate_and_print(CONFIG, valid_preds_df,
                       'Test_Set_Word-level_Predictions.csv')
    tabulate_and_print(CONFIG, train_preds_df,
                       'Train_Set_Word-level_Predictions.csv')

    print("Printing Top-k Accuracy reports")
    topk_accuracy_report(CONFIG,
                         train_preds_df,
                         valid_preds_df,
                         word_str='word1')
    topk_accuracy_report(CONFIG,
                         train_preds_df,
                         valid_preds_df,
                         word_str='word2')
    topk_accuracy_report(CONFIG,
                         train_preds_df,
                         valid_preds_df,
                         word_str='bigram')

    train_freqs = {vocab[key]: val for key, val in word2freq.items()}
    remove_tokens = [
        CONFIG["begin_token"], CONFIG["end_token"], CONFIG["oov_token"],
        CONFIG["pad_token"]
    ]

    for string in ['word1', 'word2']:
        print(f'Postprocessing for {string}')
        word_wise_roc(CONFIG,
                      vocab,
                      valid_preds_df,
                      valid_all_preds,
                      train_freqs,
                      remove_tokens,
                      i2w,
                      string=string)

    # Post-processing for bigrams as classes
    bigram_i2w, bigram_w2i = return_bigram_vocab(vocab)
    valid_preds_df['bigram_index'] = valid_preds_df.set_index(
        ['word1', 'word2']).index.map(bigram_w2i.get)

    true = np.array(valid_preds_df['bigram_index'])
    labels = np.zeros((true.size, n_classes**2))
    labels[np.arange(true.size), true] = 1
    predictions = return_bigram_proba(valid_all_preds, len(vocab))

    my_df = save_bigram_counts(CONFIG, [y_train, y_test], word2freq, i2w,
                               'bigram-counts.csv')
    raw_train_df = my_df.loc[:, ['word1', 'word2', 'count_train']]
    raw_train_df['bigram_index'] = raw_train_df.set_index(
        ['word1', 'word2']).index.map(bigram_w2i)
    bigram_train_freqs = calc_bigram_train_freqs(raw_train_df, 'count_train')

    print("Evaluating top-k")
    evaluate_topk(predictions.numpy(),
                  true,
                  bigram_i2w,
                  Counter(bigram_train_freqs),
                  CONFIG["SAVE_DIR"],
                  min_train=10,
                  prefix='bigram',
                  suffix='bigram',
                  tokens_to_remove=remove_tokens)

    print("Evaluating ROC-AUC")
    evaluate_roc(predictions,
                 labels,
                 bigram_i2w,
                 bigram_train_freqs,
                 CONFIG["SAVE_DIR"],
                 do_plot=True,
                 given_thresholds=None,
                 title='bigram',
                 suffix='bigram',
                 min_train=10,
                 tokens_to_remove=remove_tokens)

    bigram_accuracy_report(CONFIG, vocab, i2w, valid_all_trg_y,
                           valid_all_preds)

    print("Generating reports for concatenated outputs")
    bigram_predictions = concatenate_bigrams(CONFIG, valid_all_trg_y,
                                             valid_all_preds, vocab, i2w,
                                             bigram_w2i, true, n_classes)
    evaluate_roc(bigram_predictions,
                 labels,
                 bigram_i2w,
                 bigram_train_freqs,
                 CONFIG["SAVE_DIR"],
                 do_plot=True,
                 given_thresholds=None,
                 title='bigramc',
                 suffix='bigramc',
                 min_train=5,
                 tokens_to_remove=remove_tokens)

    evaluate_topk(bigram_predictions,
                  true,
                  bigram_i2w,
                  Counter(bigram_train_freqs),
                  CONFIG["SAVE_DIR"],
                  min_train=5,
                  prefix='bigramc',
                  suffix='bigramc',
                  tokens_to_remove=remove_tokens)

end_time = datetime.now()
print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
print(f'Total runtime: {end_time - start_time} (HH:MM:SS)')
