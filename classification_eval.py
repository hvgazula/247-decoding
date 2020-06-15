import json
from collections import Counter
from pprint import pprint

import numpy as np
import torch
import torch.nn as nn

from eval_utils import evaluate_roc, evaluate_topk


def report_classification(CONFIG, args, DEVICE, model, data_iterator):

    data_X = [x[0] for x in data_iterator.dataset.examples]
    data_y = [x[1] for x in data_iterator.dataset.examples]

    if args.gpus:
        if args.gpus > 1:
            model = nn.DataParallel(model)
        model.to(DEVICE)

    start, end = 0, 0
    softmax = nn.Softmax(dim=1)
    all_preds = np.zeros((data_X.size(0), n_classes), dtype=np.float32)
    print('Allocating', np.prod(all_preds.shape) * 5 / 1e9, 'GB')

    # Calculate all predictions on test set
    with torch.no_grad():
        for batch in data_iterator:
            src, _ = batch[0].to(DEVICE), batch[1].to(DEVICE, dtype=torch.long)
            end = start + src.size(0)
            out = softmax(model(src))
            all_preds[start:end, :] = out.cpu()
            start = end

    print("Calculated predictions")

    # Make categorical
    n_examples = data_y.shape[0]
    categorical = np.zeros((n_examples, n_classes), dtype=np.float32)
    categorical[np.arange(n_examples), data_y] = 1

    train_freq = Counter(y_train.tolist())

    # Evaluate top-k
    print("Evaluating top-k")
    res = evaluate_topk(all_preds,
                        data_y.numpy(),
                        i2w,
                        train_freq,
                        CONFIG["SAVE_DIR"],
                        suffix='-val',
                        min_train=args.vocab_min_freq)

    # Evaluate ROC-AUC
    print("Evaluating ROC-AUC")
    res.update(
        evaluate_roc(all_preds,
                     categorical,
                     i2w,
                     train_freq,
                     CONFIG["SAVE_DIR"],
                     do_plot=not args.no_plot,
                     min_train=args.vocab_min_freq))
    pprint(res.items())

    print("Saving results")
    with open(CONFIG["SAVE_DIR"] + "results.json", "w") as fp:
        json.dump(res, fp, indent=4)
