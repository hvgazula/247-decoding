import argparse
from logging import raiseExceptions
import os
import pickle
from datetime import datetime

import gensim.downloader as api
import numpy as np
import pandas as pd
import torch
# from transformers import (BartForConditionalGeneration, BartTokenizer,
#                           BertForMaskedLM, BertTokenizer, RobertaForMaskedLM,
#                           RobertaTokenizer)
import torch.utils.data as data
from transformers import BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer


def save_pickle(item, file_name):
    """Write 'item' to 'file_name.pkl'
    """
    if '.pkl' not in file_name:
        add_ext = '.pkl'
    else:
        add_ext = ''

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


def tokenize_and_explode(df, tokenizer):
    """Tokenizes the words/labels and creates a row for each token

    Args:
        df (DataFrame): dataframe of labels
        tokenizer (tokenizer): from transformers

    Returns:
        DataFrame: a new dataframe object with the words tokenized
    """
    df['token'] = df.word.apply(tokenizer.tokenize)
    df = df.explode('token', ignore_index=True)

    return df


def create_token_ids(args, utter_datum, tokenizer):
    tokens = utter_datum['token'].values
    # Convert to numerical indices
    token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokens),
                             device=args.device)
    assert len(tokens) == len(token_ids)
    return tokens, token_ids


def load_pretrained_model(args):
    # Load pre-trained model
    model = args.model_class.from_pretrained(args.model_name,
                                             local_files_only=False,
                                             output_hidden_states=True)
    model = model.to(args.device)
    model.eval()  # evaluation mode to deactivate the DropOut modules


def setup_environ(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    base_name = f'{args.model_name}-c-{args.context_length}-{args.suffix}'
    os.makedirs('results-predictions/' + base_name, exist_ok=True)

    args.base_name = base_name
    args.device = device
    return


def select_token_model(args):

    args.pickle_name = args.subject + '_labels.pkl'

    if 'roberta' in args.model_name:
        tokenizer_class = RobertaTokenizer
        model_class = RobertaForMaskedLM
    elif 'bert' in args.model_name:
        tokenizer_class = BertTokenizer
        model_class = BertForMaskedLM
    elif 'bart' in args.model_name:
        tokenizer_class = BartTokenizer
        model_class = BartForConditionalGeneration
    else:
        print('No model found for', args.model_name)
        exit(1)

    args.tokenizer_class = tokenizer_class
    args.model_class = model_class

    tokenizer = tokenizer_class.from_pretrained(args.model_name)
    if args.context_length <= 0:
        args.context_length = tokenizer.max_len
    assert args.context_length <= tokenizer.max_len, \
        'given length is greater than max length'

    return tokenizer


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
    parser.add_argument('--subject', type=str, default='676')
    parser.add_argument('--history', action='store_true', default=False)

    custom_args = [
        '--subject', '625', '--history', '--save-hidden-states',
        '--embedding-type', 'bert'
    ]

    args = parser.parse_args(custom_args)
    return args


def build_context(args, tokenizer):
    context_len = args.context_length - 3  # [CLS], [SEP] hist/sent, [SEP] end

    # add [CLS] + history + [SEP]
    if args.history:
        start, end = max(0, sentence_idxs[0] + len(sentence) - context_len), \
            sentence_idxs[0]
        history = token_ids[start:end]
        # remove history from other conversations
        history = history[np.where(
            pd.to_numeric(utter_datum['conversation_id'][start:end]) ==
            conversationID)[0]]
        context = [tokenizer.cls_token_id] + history.tolist() + \
                  [tokenizer.sep_token_id] + sentence.tolist() + \
                  [tokenizer.sep_token_id]

    else:  # add [CLS] (no history, just sentence, default)
        context = [tokenizer.cls_token_id] +  \
                   sentence.tolist() + [tokenizer.sep_token_id]

    context = torch.tensor(context, device=device).unsqueeze(0)

    return context


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


def map_embeddings_to_tokens(df, embed):

    # df = df.reindex(columns=[*df.columns.tolist(), 'embeddings'])
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


def gen_bert_embeddings(args, df):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    unique_sentence_list = get_unique_sentences(df)
    df = tokenize_and_explode(df, tokenizer)

    tokens = tokenizer.batch_encode_plus(unique_sentence_list,
                                         padding=True,
                                         return_tensors='pt')
    input_ids_val = tokens['input_ids']
    attention_masks_val = tokens['attention_mask']

    dataset = data.TensorDataset(input_ids_val, attention_masks_val)
    data_dl = data.DataLoader(dataset, batch_size=256, shuffle=True)

    concat_output = []
    with torch.no_grad():
        model = model.to(device)
        model.eval()
        for i, batch in enumerate(data_dl):
            batch = tuple(b.to(device) for b in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
            }
            model_output = model(**inputs)
            # The last hidden-state is the first element of the output tuple
            print(model_output[0].shape)
            concat_output.append(model_output[0].detach().cpu().numpy())
    embeddings = np.concatenate(concat_output, axis=0)

    print(embeddings.shape)
    emb_df = map_embeddings_to_tokens(df, embeddings)
    save_pickle(emb_df.to_dict('records'), '625_bert_embeddings')

    return


def gen_gpt2_embeddings(args, df):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    unique_sentence_list = get_unique_sentences(df)
    df = tokenize_and_explode(df, tokenizer)

    tokens = tokenizer.batch_encode_plus(unique_sentence_list, padding=True)
    input_ids = torch.tensor(tokens)

    batch_size = 256
    data_dl = data.DataLoader(input_ids, batch_size=batch_size, shuffle=True)

    concat_output = []
    with torch.no_grad():
        model = model.to(device)
        model.eval()
        for i, batch in enumerate(data_dl):
            batch = batch.to(device)
            model_output = model(batch)
            # The last hidden-state is the first element of the output tuple
            print(model_output[0][-1].shape)
            concat_output.append(model_output[0].detach().cpu().numpy())
    embeddings = np.concatenate(concat_output, axis=0)

    print(embeddings.shape)
    emb_df = map_embeddings_to_tokens(df, embeddings)
    save_pickle(emb_df.to_dict('records'), '625_gpt2_embeddings')

    return


def main():

    args = parse_arguments()
    setup_environ(args)

    # tokenizer = select_token_model(args)
    # load_pretrained_model(args)

    args.pickle_name = args.subject + '_labels.pkl'
    utter_orig = load_pickle(args.pickle_name)

    if args.embedding_type == 'glove':
        gen_word2vec_embeddings(args, utter_orig)
    elif args.embedding_type == 'bert':
        gen_bert_embeddings(args, utter_orig)
    elif args.embedding_type == 'gpt2':
        gen_gpt2_embeddings(args, utter_orig)

    return


if __name__ == '__main__':
    start_time = datetime.now()
    print(f'Start Time: {start_time.strftime("%A %m/%d/%Y %H:%M:%S")}')

    main()

    end_time = datetime.now()
    print(f'End Time: {end_time.strftime("%A %m/%d/%Y %H:%M:%S")}')
    print(f'Total runtime: {end_time - start_time} (HH:MM:SS)')
