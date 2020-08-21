import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def classify_neural_signal(CONFIG, vocab, device, model, data_iterator):
    start, end = 0, 0
    softmax = nn.Softmax(dim=1)
    all_preds = np.zeros((len(data_iterator.dataset), len(vocab)),
                         dtype=np.float32)
    print('Allocating', np.prod(all_preds.shape) * 5 / 1e9, 'GB')

    # Calculate all predictions on test set
    with torch.no_grad():
        model.eval()
        for batch in data_iterator:
            src = batch[0].to(device)
            end = start + src.size(0)
            out = model(src)
            if src.shape[0] == 1:
                out = out.view(1, -1)
            out = softmax(out)
            all_preds[start:end, :] = out.cpu()
            start = end

    topk_preds_scores, topk_preds = torch.topk(torch.tensor(all_preds), 10)
    _, all_trg_y = zip(*data_iterator.dataset.examples)

    return all_trg_y, topk_preds, topk_preds_scores, all_preds


def word_pred_scores(valid_all_preds, valid_preds_df, w2i, string=None):
    dictList1 = []

    string = check_string(string)

    for row_np, row_df in zip(valid_all_preds, valid_preds_df.iterrows()):
        word1_row = [
            row for i, row in row_df[1].iteritems() if i.startswith(string)
        ]
        d1 = {k: 0 for k in w2i}
        for key, value in zip(word1_row, row_np):
            d1[key] += value
        dictList1.append(d1)
    word1_df = pd.DataFrame(dictList1)

    return word1_df


def check_string(string=None):
    """Check String Validity

    Args:
        string ([type], optional): [description]. Defaults to None.

    Raises:
        Exception: [description]

    Returns:
        str: appends '_' to input string
    """
    if not string:
        raise Exception("Wrong String")
    else:
        return string + '_'
