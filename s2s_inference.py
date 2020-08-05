import numpy as np
import torch
import torch.nn as nn


def translate_neural_signal(CONFIG, vocab, device, model, data_iterator):
    """Inference mode to translate neural signals to corresponding labels

    Args:
        CONFIG (dict): Configuration dictionary
        vocab (dict): vocabulary
        device (device): device object 'cpu' or 'gpu'
        model (model): the best model saved during training
        data_iterator (DataLoader): DataLoader

    Returns:
        all_trg_y (torch.tensor): actual target outputs
        topk_preds (torch.tensor): topk predictions
        topk_preds_scores (torch.tensor): prediction scores (between 0 and 1)
        all_preds (torch.tensor): all predictions
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


def classify_neural_signal(CONFIG, vocab, device, model, data_iterator):
    if CONFIG["gpus"]:
        if CONFIG["gpus"] > 1:
            model = nn.DataParallel(model)
        model.to(device)

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
