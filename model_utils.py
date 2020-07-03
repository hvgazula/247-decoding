import os
import sys

import torch

from models import *


def return_model(CONFIG, vocab):
    """Build selected model

    Args:
        CONFIG (dict): configuration info
        vocab (dict): vocabulary

    Returns:
        torch.model: PITOM/ConvNet10/MeNTALmini/MeNTAL
    """
    # Default models and parameters
    DEFAULT_MODELS = {
        "ConvNet10": (len(vocab), ),
        "PITOM": (len(vocab), sum(CONFIG["max_electrodes"])),
        "MeNTALmini":
        (sum(CONFIG["max_electrodes"]), len(vocab), CONFIG["tf_dmodel"],
         CONFIG["tf_nhead"], CONFIG["tf_nlayer"], CONFIG["tf_dff"],
         CONFIG["tf_dropout"]),
        "MeNTAL": (sum(CONFIG["max_electrodes"]), len(vocab),
                   CONFIG["tf_dmodel"], CONFIG["tf_nhead"],
                   CONFIG["tf_nlayer"], CONFIG["tf_dff"], CONFIG["tf_dropout"])
    }

    # Create model
    if CONFIG["init_model"] is None:
        if CONFIG["model"] in DEFAULT_MODELS:
            print("Building default model: %s" % CONFIG["model"], end="")
            model_class = globals()[CONFIG["model"]]
            model = model_class(*(DEFAULT_MODELS[CONFIG["model"]]))
        else:
            print("Building custom model: %s" % CONFIG["model"], end="")
            sys.exit(1)
    else:
        model_name = "%s%s.pt" % (CONFIG["SAVE_DIR"], CONFIG["model"])
        if os.path.isfile(model_name):
            model = torch.load(model_name)
            model = model.module if hasattr(model, 'module') else model
            print("Loaded initial model: %s " % CONFIG["model"])
        else:
            print("No models found in: ", CONFIG["SAVE_DIR"])
            sys.exit(1)
    print(" with %d trainable parameters" %
          sum([p.numel() for p in model.parameters() if p.requires_grad]))
    sys.stdout.flush()

    return model
