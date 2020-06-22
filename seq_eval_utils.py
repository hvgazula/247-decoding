import itertools
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from eval_utils import evaluate_roc, evaluate_topk


def translate_neural_signal(CONFIG, vocab, device, model, data_iterator):
    """[summary]

    Args:
        CONFIG (dict): Configuration dictionary
        vocab (dict): vocabulary
        device (device): device object 'cpu' or 'gpu'
        model (model): the best model saved during training
        data_iterator (DataLoader): DataLoader

    Returns:
        [type]: [description]
    """
    vocab_len = len(vocab)
    data_set_len = len(data_iterator.dataset)
    valid_bi_preds = torch.zeros(data_set_len, 3, vocab_len)
    all_trg_y = torch.zeros(data_set_len, 3, dtype=torch.int32)

    softmax = nn.Softmax(dim=1)

    if CONFIG["gpus"]:
        model.to(device)

    # Calculate all predictions on test set
    with torch.no_grad():
        model.eval()

        for enum, batch in enumerate(data_iterator):

            src = batch[0].to(device)
            trg_y = batch[2].long().to(device)
            trg_pos_mask = batch[3].to(device).squeeze()
            trg_pad_mask = batch[4].to(device)

            all_trg_y[enum * CONFIG["batch_size"]:(enum + 1) *
                      CONFIG["batch_size"], :] = trg_y

            memory = model.encode(src)
            y = torch.zeros(src.size(0), 1, len(vocab)).long().to(device)
            y[:, :, vocab[CONFIG["begin_token"]]] = 1

            bi_out = torch.zeros(len(batch[0]), trg_y.shape[1], len(vocab))
            for i in range(trg_y.size(1)):
                out = model.decode(memory, y,
                                   trg_pos_mask[:y.size(1), :y.size(1)],
                                   trg_pad_mask[:, :y.size(1)])[:, -1, :]
                out = softmax(out / CONFIG["temp"])
                bi_out[:, i, :] = out
                temp = torch.zeros(src.size(0), vocab_len).long().to(device)
                temp = temp.scatter_(1,
                                     torch.argmax(out, dim=1).unsqueeze(-1), 1)
                y = torch.cat([y, temp.unsqueeze(1)], dim=1)
            valid_bi_preds[enum * CONFIG["batch_size"]:(enum + 1) *
                           CONFIG["batch_size"], :, :] = bi_out

        topk_preds_scores = torch.topk(valid_bi_preds, 10).values
        topk_preds = torch.topk(valid_bi_preds, 10).indices

        topk_preds_scores = topk_preds_scores.view(data_set_len, -1)
        topk_preds = topk_preds.view(data_set_len, -1)
        all_preds = valid_bi_preds.view(data_set_len, -1)

    return all_trg_y, topk_preds, topk_preds_scores, all_preds


def calc_rank(x, string):
    word = x[string]
    string = string + '_0*'
    word_preds = x.filter(regex=string)

    try:
        rank = np.where(word == word_preds)[0][0]
    except IndexError:
        rank = pd.NA

    return rank


def calc_rank1(x, string, all_preds):
    num_preds = all_preds.shape[-1] // 3
    ranks = []
    if string == 'word1':
        pred_ranks = torch.argsort(all_preds[:, :num_preds],
                                   dim=1,
                                   descending=True)
    elif string == 'word2':
        pred_ranks = torch.argsort(all_preds[:, num_preds:2 * num_preds],
                                   dim=1,
                                   descending=True)

    for i, word in enumerate(x):
        ranks.append(np.where(word == pred_ranks[i])[0][0])
    return ranks


def fill_topk_cols(x, string):
    rank = x['_'.join([string, 'rank'])]

    if pd.isna(rank):
        abc = [0, 0, 0]
    elif rank == 0:
        abc = [1, 0, 0]
    elif rank < 5:
        abc = [0, 1, 0]
    elif rank < 10:
        abc = [0, 0, 1]
    else:
        abc = [0, 0, 0]

    return abc


def apply_rank(df, all_preds, string=None):
    if not str:
        print("Bad String")
        return 0

    rank_word = '_'.join([string, 'rank'])
    top_col_names = ['_'.join([string, 't' + str(i)]) for i in [1, 5, 10]]

    # df[rank_word] = df.apply(calc_rank, axis=1, args=(string, ))
    df[rank_word] = calc_rank1(df[string].tolist(), string, all_preds)
    df[top_col_names] = pd.DataFrame(
        df.apply(fill_topk_cols, axis=1, args=(string, )).tolist())

    return df


def create_excel_preds(targets, top_predictions, all_preds, i2w):
    df = pd.DataFrame(targets.numpy(), columns=['word1', 'word2', 'word3'])
    df = df.drop(columns=['word3'])
    pred_col_names = [
        '_'.join([word, str(i).zfill(2)]) for word in ['word1', 'word2']
        for i in range(1, 11)
    ]

    df = apply_rank(df, all_preds, string='word1')
    df = apply_rank(df, all_preds, string='word2')

    df[pred_col_names] = pd.DataFrame(top_predictions.numpy()[:, :20])

    df[pred_col_names] = df[pred_col_names].replace(i2w)
    df['word1'] = df['word1'].replace(i2w)
    df['word2'] = df['word2'].replace(i2w)

    return df


def replace_words(data, i2w):
    df_y_train = pd.DataFrame(data,
                              columns=['start', 'word1', 'word2', 'stop'])
    df_y_train['word1'].replace(i2w, inplace=True)
    df_y_train['word2'].replace(i2w, inplace=True)

    return df_y_train


def bigram_freq_excel(CONFIG, data, word2freq, i2w, filename, ref_data=None):
    valid_df = replace_words(data, i2w)
    valid_df = valid_df.groupby(['word1',
                                 'word2']).size().reset_index(name='Count')
    valid_df['BF1'] = valid_df['word1'].replace(
        dict(valid_df['word1'].value_counts()))
    valid_df['BF2'] = valid_df['word2'].replace(
        dict(valid_df['word2'].value_counts()))
    valid_df['VF1'] = valid_df['word1'].replace(word2freq)
    valid_df['VF2'] = valid_df['word2'].replace(word2freq)

    if ref_data is not None:
        valid_df = valid_df.merge(ref_data,
                                  on=['word1', 'word2'],
                                  suffixes=('_valid', '_train'),
                                  how='left')

    valid_df.to_excel(os.path.join(CONFIG["SAVE_DIR"], filename), index=False)

    # print(len(valid_df['word1'].unique()))
    # print(len(valid_df['word2'].unique()))

    # print(set(word2freq.keys()) - set(valid_df['word1'].unique()))
    # print(set(word2freq.keys()) - set(valid_df['word2'].unique()))

    return valid_df


def word_wise_roc(CONFIG,
                  vocab,
                  valid_preds_df,
                  valid_all_preds,
                  train_freqs,
                  remove_tokens,
                  i2w,
                  string=None):
    n_classes = len(vocab)

    if string == 'word1':
        col_range = range(n_classes * 0, n_classes * 1)
    elif string == 'word2':
        col_range = range(n_classes * 1, n_classes * 2)
    else:
        sys.exit('Wrong Word')

    true = np.array(valid_preds_df[string].replace(vocab).tolist())
    labels = np.zeros((true.size, true.max() + 1))
    labels[np.arange(true.size), true] = 1
    predictions = valid_all_preds.numpy()[:, col_range]
    evaluate_topk(predictions,
                  true,
                  i2w,
                  Counter(train_freqs),
                  CONFIG["SAVE_DIR"],
                  prefix=string,
                  suffix=string,
                  min_train=10,
                  tokens_to_remove=remove_tokens)

    evaluate_roc(predictions,
                 labels,
                 i2w,
                 train_freqs,
                 CONFIG["SAVE_DIR"],
                 do_plot=True,
                 given_thresholds=None,
                 title=string,
                 suffix=string,
                 min_train=10,
                 tokens_to_remove=remove_tokens)
    return


def return_bigram_proba(preds, n_classes):
    # EXAMPLE
    # a = np.array([[1,2],[3,4]])
    # np.repeat(a, 2, axis=2)
    # np.tile(a, (1, 2))
    softmax = nn.Softmax(dim=1)

    first = preds[:, n_classes * 0:n_classes * 1]
    second = preds[:, n_classes * 1:n_classes * 2]

    first_repeat = torch.repeat_interleave(
        first, n_classes, dim=1)  # this is similar to np.repeat
    second_repeat = second.repeat((1, n_classes))  # this is similar to np.tile

    all_preds = first_repeat * second_repeat  # hadamard product
    all_preds = softmax(all_preds)  # softmax in dim=1

    return all_preds


def return_bigram_vocab(vocab):
    abc = [p for p in itertools.product(vocab.keys(), repeat=2)]
    w2i = {word: i for i, word in enumerate(abc)}
    i2w = {i: '_'.join(words) for i, words in enumerate(abc)}
    return i2w, w2i


def calc_bigram_train_freqs(df):
    a = df[['bigram_index', 'Count']]
    kabc = a.to_dict('records')
    train_freqs = {item['bigram_index']: item['Count'] for item in kabc}
    return train_freqs
