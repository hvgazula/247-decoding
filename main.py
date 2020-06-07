import sys
from datetime import datetime

import torch
from sklearn.model_selection import train_test_split

from arg_parser import arg_parser
from build_matrices import build_design_matrices_seq2seq
from config import build_config
from filter_utils import filter_by_labels, filter_by_signals
from plot_utils import figure5
from rw_utils import bigram_counts_to_csv
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

    print('Plotting Distribution of Signal Lengths')
    seq_lengths = [seq.shape[0] for seq in signals]
    figure5(CONFIG["SAVE_DIR"], seq_lengths, 'all')

    signals, labels = filter_by_signals(signals, labels, 75)
    signals, labels = filter_by_labels(signals, labels, 30)

    bigram_counts_to_csv(CONFIG, labels, 'mixed')

    X_train, X_test, y_train, y_test = train_test_split(signals,
                                                        labels,
                                                        stratify=labels,
                                                        test_size=0.30,
                                                        random_state=args.seed)

    bigram_counts_to_csv(CONFIG, y_train, 'train')
    bigram_counts_to_csv(CONFIG, y_test, 'test')
