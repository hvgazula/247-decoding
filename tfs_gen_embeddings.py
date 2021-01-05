import argparse
import os
import pickle
from datetime import datetime
from itertools import islice

import gensim.downloader as api
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from transformers import (BartForConditionalGeneration, BartTokenizer,
                          BertForMaskedLM, BertTokenizer, GPT2LMHeadModel,
                          GPT2Tokenizer, RobertaForMaskedLM, RobertaTokenizer)


def save_pickle(item, file_name):
    """Write 'item' to 'file_name.pkl'
    """
    add_ext = '' if file_name.endswith('.pkl') else '.pkl'

    file_name = os.path.join(os.getcwd(), 'pickles', file_name) + add_ext
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, 'wb') as fh:
        pickle.dump(item, fh)
    return


def load_pickle(file):
    """Load the datum pickle and returns as a dataframe

    Args:
        file (string): labels pickle from 247-decoding/tfs_pickling.py

    Returns:
        DataFrame: pickle contents returned as dataframe
    """
    with open(file, 'rb') as fh:
        datum = pickle.load(fh)

    df = pd.DataFrame.from_dict(datum['labels'])

    return df


def tokenize_and_explode(args, df, tokenizer):
    """Tokenizes the words/labels and creates a row for each token

    Args:
        df (DataFrame): dataframe of labels
        tokenizer (tokenizer): from transformers

    Returns:
        DataFrame: a new dataframe object with the words tokenized
    """
    df['token'] = df.word.apply(tokenizer.tokenize)
    df = df.explode('token', ignore_index=True)
    df['token_id'] = df['token'].apply(tokenizer.convert_tokens_to_ids)

    if args.embedding_type == 'gpt2':
        df['gpt2_token_is_root'] = chr(288) + df['word'] == df['token']
    elif args.embedding_type == 'bert':
        df['bert_token_is_root'] = df['word'] == df['token']

    return df


def map_embeddings_to_tokens(df, embed):

    multi = df.set_index(['conversation_id', 'sentence_idx', 'sentence'])
    unique_sentence_idx = multi.index.unique().values

    uniq_sentence_count = len(get_unique_sentences(df))
    assert uniq_sentence_count == len(embed)

    c = []
    for unique_idx, sentence_embedding in zip(unique_sentence_idx, embed):
        a = df['conversation_id'] == unique_idx[0]
        b = df['sentence_idx'] == unique_idx[1]
        num_tokens = sum(a & b)
        c.append(pd.Series(sentence_embedding[1:num_tokens + 1, :].tolist()))

    df['embeddings'] = pd.concat(c, ignore_index=True)
    return df


def get_unique_sentences(df):
    return df[['conversation_id', 'sentence_idx',
               'sentence']].drop_duplicates()['sentence'].tolist()


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) <= n:
        yield result
    for elem in it:
        result = result[1:] + (elem, )
        yield result


def extract_token_embeddings(concat_output):
    """(batch_size, max_len, embedding_size)"""
    # concatenate all batches
    concatenated_embeddings = np.concatenate(concat_output, axis=0)
    if concatenated_embeddings.shape[0] == 1:
        return np.squeeze(concatenated_embeddings, axis=0)

    # the first token is always empty
    init_token_embedding = np.empty((1, 768)) * np.nan

    # From the first example take all embeddings except the last one
    first_window_all_tokens = concatenated_embeddings[0, :-1, :]

    # From all other examples take the penultimate embeddings
    other_windows_last_token = concatenated_embeddings[1:, -2, :]

    extracted_embeddings = np.concatenate([
        init_token_embedding, first_window_all_tokens, other_windows_last_token
    ],
                                          axis=0)

    return extracted_embeddings


def generate_embeddings_with_context(args, df):
    tokenizer = args.tokenizer
    model = args.model
    device = args.device

    if args.gpus > 1:
        model = nn.DataParallel(model)

    df = tokenize_and_explode(args, df, tokenizer)

    if args.embedding_type == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token

    final_embeddings = []
    for conversation in df.conversation_id.unique():
        token_list = df[df.conversation_id ==
                        conversation]['token_id'].tolist()
        sliding_windows = list(window(token_list, 1024))
        print(
            f'conversation: {conversation}, tokens: {len(token_list)}, #sliding: {len(sliding_windows)}'
        )
        input_ids = torch.tensor(sliding_windows)
        data_dl = data.DataLoader(input_ids, batch_size=16, shuffle=True)

        with torch.no_grad():
            model = model.to(device)
            model.eval()

            concat_output = []
            for i, batch in enumerate(data_dl):
                print(f'batch: {i}')
                batch = batch.to(args.device)
                model_output = model(batch)
                concat_output.append(
                    model_output[-1][-1].detach().cpu().numpy())

        extracted_embeddings = extract_token_embeddings(concat_output)
        assert extracted_embeddings.shape[0] == len(token_list)
        final_embeddings.append(extracted_embeddings)

    df['embeddings'] = np.concatenate(final_embeddings, axis=0).tolist()
    # TODO: convert embeddings dtype from object to float

    output_file = '_'.join(
        [args.subject, args.embedding_type, 'contextual_embeddings'])
    save_pickle(df.to_dict('records'), output_file)

    return df


def generate_embeddings(args, df):
    tokenizer = args.tokenizer
    model = args.model
    device = args.device

    df = tokenize_and_explode(args, df, tokenizer)
    unique_sentence_list = get_unique_sentences(df)

    if args.embedding_type == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token

    tokens = tokenizer(unique_sentence_list, padding=True, return_tensors='pt')
    input_ids_val = tokens['input_ids']
    attention_masks_val = tokens['attention_mask']

    dataset = data.TensorDataset(input_ids_val, attention_masks_val)
    data_dl = data.DataLoader(dataset, batch_size=16, shuffle=True)

    with torch.no_grad():
        model = model.to(device)
        model.eval()

        concat_output = []
        for batch in data_dl:
            batch = tuple(b.to(device) for b in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
            }
            model_output = model(**inputs)
            concat_output.append(model_output[-1][-1].detach().cpu().numpy())

    embeddings = np.concatenate(concat_output, axis=0)
    emb_df = map_embeddings_to_tokens(df, embeddings)
    output_file = '_'.join([args.subject, args.embedding_type, 'embeddings'])
    save_pickle(emb_df.to_dict('records'), output_file)

    return


def get_vector(x, glove):
    try:
        return glove.get_vector(x)
    except KeyError:
        return None


def gen_word2vec_embeddings(args, df):
    glove = api.load('glove-wiki-gigaword-50')
    df['embeddings'] = df['word'].apply(lambda x: get_vector(x, glove))
    save_pickle(df.to_dict('records'), '625_glove50_embeddings')
    return


def setup_environ(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.pickle_name = args.subject + '_labels.pkl'
    args.gpus = torch.cuda.device_count()

    if args.gpus > 1:
        args.model = nn.DataParallel(args.model)
    return


def select_tokenizer_and_model(args):

    if args.embedding_type == 'gpt2':
        tokenizer_class = GPT2Tokenizer
        model_class = GPT2LMHeadModel
        model_name = 'gpt2'
    elif args.embedding_type == 'roberta':
        tokenizer_class = RobertaTokenizer
        model_class = RobertaForMaskedLM
        model_name = 'roberta'
    elif args.embedding_type == 'bert':
        tokenizer_class = BertTokenizer
        model_class = BertForMaskedLM
        model_name = 'bert-large-uncased-whole-word-masking'
    elif args.embedding_type == 'bart':
        tokenizer_class = BartTokenizer
        model_class = BartForConditionalGeneration
        model_name = 'bart'
    else:
        print('No model found for', args.model_name)
        exit(1)

    args.model = model_class.from_pretrained(model_name,
                                             output_hidden_states=True)
    args.tokenizer = tokenizer_class.from_pretrained(model_name,
                                                     add_prefix_space=True)

    # if args.context_length <= 0:
    #     args.context_length = args.tokenizer.max_len
    # assert args.context_length <= args.tokenizer.max_len, \
    #     'given length is greater than max length'

    return


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name',
                        type=str,
                        default='bert-large-uncased-whole-word-masking')
    parser.add_argument('--embedding-type', type=str, default='glove')
    parser.add_argument('--context-length', type=int, default=512)
    parser.add_argument('--save-predictions',
                        action='store_true',
                        default=False)
    parser.add_argument('--save-hidden-states',
                        action='store_true',
                        default=False)
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--subject', type=str, default='625')
    parser.add_argument('--history', action='store_true', default=False)

    return parser.parse_args()


def main():
    args = parse_arguments()
    select_tokenizer_and_model(args)
    setup_environ(args)
    utterance_df = load_pickle(args.pickle_name)

    if args.history:
        if args.embedding_type == 'gpt2':
            generate_embeddings_with_context(args, utterance_df)
        else:
            print('TODO: Generate embeddings for this model with context')
        return

    if args.embedding_type == 'glove':
        gen_word2vec_embeddings(args, utterance_df)
    else:
        generate_embeddings(args, utterance_df)

    return


if __name__ == '__main__':
    start_time = datetime.now()
    print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

    main()

    end_time = datetime.now()
    print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
    print(f'Total runtime: {end_time - start_time} (HH:MM:SS)')
