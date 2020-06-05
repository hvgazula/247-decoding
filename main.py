import sys
from datetime import datetime

import torch

from arg_parser import arg_parser
from build_matrices import build_design_matrices_seq2seq
from config import build_config
from utils import fix_random_seed

now = datetime.now()
date_str = now.strftime("%A %d/%m/%Y %H:%M:%S")
results_str = now.strftime("%Y-%m-%d-%H:%M")

args = arg_parser()
CONFIG = build_config(args, results_str)
sys.stdout = open(CONFIG["LOG_FILE"], 'w')

print(f'Start Time: {date_str}')
print(f'Setting Random seed: {CONFIG["seed"]}')
fix_random_seed(CONFIG)

# Model objectives
MODEL_OBJ = {
    "ConvNet10": "classifier",
    "PITOM": "classifier",
    "MeNTALmini": "classifier",
    "MeNTAL": "seq2seq"
}

# GPUs
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
args.gpus = min(args.gpus, torch.cuda.device_count())

args.model = args.model.split("_")[0]
classify = False if (args.model in MODEL_OBJ
                     and MODEL_OBJ[args.model] == "seq2seq") else True

if classify:
    pass
else:
    signals, labels = build_design_matrices_seq2seq(CONFIG,
                                                    delimiter=" ",
                                                    aug_shift_ms=[-1000, -500],
                                                    max_num_bins=0)
