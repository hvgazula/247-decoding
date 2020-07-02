import argparse
from typing import List, Optional


def arg_parser(default_args: Optional[List] = None):
    '''Read arguments from the command line

    Args:
        default_args: None/List of arguments (seeexamples)

    Examples::
        >>> output = arg_parser()
        >>> output = arg_parser(['--model', 'PITOM',
                                '--subjects', '625', '676'])

    Miscellaneous:
        model: DNN model to choose from (PITOM, ConvNet, MeNTALmini, MeNTAL)
        subjects: (list of strings): subject id's as a list
        shift (integer): Amount by which the onset should be shifted
        lr (float): learning rate
        gpus (int): number of gpus for the model to run on
        epochs (int): number of epochs
        batch-size (int): bach-size
        window-size (int): window size to consider for the word in ms
        bin-size (int): bin size in milliseconds
        ...and so on
'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MeNTAL')
    parser.add_argument('--subjects', nargs='*', default=['625'])
    parser.add_argument('--shift', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpus', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--window-size', type=int, default=2000)
    parser.add_argument('--bin-size', type=int, default=50)
    parser.add_argument('--init-model', type=str, default=None)
    parser.add_argument('--no-plot', action='store_false', default=False)
    parser.add_argument('--max-electrodes', nargs='*', type=int, default=[55])
    parser.add_argument('--vocab-min-freq', type=int, default=10)
    parser.add_argument('--vocab-max-freq', type=int, default=1000000)
    parser.add_argument('--max-num-bins', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--shuffle', action="store_true", default=False)
    parser.add_argument('--no-eval', action="store_true", default=False)
    parser.add_argument('--temp', type=float, default=0.995)
    parser.add_argument('--tf-dmodel', type=int, default=64)
    parser.add_argument('--tf-dff', type=int, default=128)
    parser.add_argument('--tf-nhead', type=int, default=4)
    parser.add_argument('--tf-nlayer', type=int, default=3)
    parser.add_argument('--tf-dropout', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=0.35)

    if not default_args:
        args = parser.parse_args()
    else:
        args = parser.parse_args(default_args)

    return args
