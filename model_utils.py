import os
import sys

import torch

from models import *


def return_model(args, CONFIG, vocab):
    # Default models and parameters
    DEFAULT_MODELS = {
        "ConvNet10": (len(vocab), ),
        "PITOM": (len(vocab), sum(args.max_electrodes)),
        "MeNTALmini":
        (sum(args.max_electrodes), len(vocab), args.tf_dmodel, args.tf_nhead,
         args.tf_nlayer, args.tf_dff, args.tf_dropout),
        "MeNTAL": (sum(args.max_electrodes), len(vocab), args.tf_dmodel,
                   args.tf_nhead, args.tf_nlayer, args.tf_dff, args.tf_dropout)
    }

    # Create model
    if args.init_model is None:
        if args.model in DEFAULT_MODELS:
            print("Building default model: %s" % args.model, end="")
            model_class = globals()[args.model]
            model = model_class(*(DEFAULT_MODELS[args.model]))
        else:
            print("Building custom model: %s" % args.model, end="")
            sys.exit(1)
    else:
        model_name = "%s%s.pt" % (CONFIG["SAVE_DIR"], args.model)
        if os.path.isfile(model_name):
            model = torch.load(model_name)
            model = model.module if hasattr(model, 'module') else model
            print("Loaded initial model: %s " % args.model)
        else:
            print("No models found in: ", CONFIG["SAVE_DIR"])
            sys.exit(1)
    print(" with %d trainable parameters" %
          sum([p.numel() for p in model.parameters() if p.requires_grad]))
    sys.stdout.flush()

    return model
