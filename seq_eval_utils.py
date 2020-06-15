import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def translate_neural_signal(CONFIG, args, vocab, device, model, data_iterator,
                            vocab_len):

    data_set_len = len(data_iterator.dataset)
    valid_bi_preds = torch.zeros(data_set_len, 3, vocab_len)
    all_trg_y = torch.zeros(data_set_len, 3, dtype=torch.int32)

    if args.gpus:
        model.to(device)

    softmax = nn.Softmax(dim=1)

    # Calculate all predictions on test set
    with torch.no_grad():
        model.eval()

        for enum, batch in enumerate(data_iterator):

            src = batch[0].to(device)
            trg_y = batch[2].long().to(device)
            trg_pos_mask = batch[3].to(device).squeeze()
            trg_pad_mask = batch[4].to(device)

            all_trg_y[enum * args.batch_size:(enum + 1) *
                      args.batch_size, :] = trg_y

            memory = model.encode(src)
            y = torch.zeros(src.size(0), 1, len(vocab)).long().to(device)
            y[:, :, vocab[CONFIG["begin_token"]]] = 1

            bi_out = torch.zeros(len(batch[0]), trg_y.shape[1], len(vocab))
            for i in range(trg_y.size(1)):
                out = model.decode(memory, y,
                                   trg_pos_mask[:y.size(1), :y.size(1)],
                                   trg_pad_mask[:, :y.size(1)])[:, -1, :]
                out = softmax(out / args.temp)
                bi_out[:, i, :] = out
                temp = torch.zeros(src.size(0), vocab_len).long().to(device)
                temp = temp.scatter_(1,
                                     torch.argmax(out, dim=1).unsqueeze(-1), 1)
                y = torch.cat([y, temp.unsqueeze(1)], dim=1)
            valid_bi_preds[enum * args.batch_size:(enum + 1) *
                           args.batch_size, :, :] = bi_out

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


def create_excel_preds(targets, top_predictions, i2w):
    df = pd.DataFrame(targets.numpy(), columns=['word1', 'word2', 'word3'])
    df = df.drop(columns=['word3'])
    pred_col_names = [
        '_'.join([word, str(i).zfill(2)]) for word in ['word1', 'word2']
        for i in range(1, 11)
    ]
    df[pred_col_names] = pd.DataFrame(top_predictions.numpy()[:, :20])
    # top_col_names = [
    #     '_'.join([word, 't' + str(i)]) for word in ['word1', 'word2']
    #     for i in [1, 5, 10]
    # ]

    df['word1_rank'] = df.apply(calc_rank, axis=1, args=('word1', ))
    df['word2_rank'] = df.apply(calc_rank, axis=1, args=('word2', ))

    df[['word1_top1', 'word1_top5', 'word1_top10']] = pd.DataFrame(
        df.apply(fill_topk_cols, axis=1, args=('word1', )).tolist())
    df[['word2_top1', 'word2_top5', 'word2_top10']] = pd.DataFrame(
        df.apply(fill_topk_cols, axis=1, args=('word2', )).tolist())

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

    print(len(valid_df['word1'].unique()))
    print(len(valid_df['word2'].unique()))

    print(set(word2freq.keys()) - set(valid_df['word1'].unique()))
    print(set(word2freq.keys()) - set(valid_df['word2'].unique()))

    return valid_df
