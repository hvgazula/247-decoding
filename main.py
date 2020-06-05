from arg_parser import arg_parser
from config import build_config
from datetime import datetime
import sys

now = datetime.now()
date_str = now.strftime("%A %d/%m/%Y %H:%M:%S")
results_str = now.strftime("%Y-%m-%d-%H:%M")

args = arg_parser()
CONFIG = build_config(args, results_str)

sys.stdout = open(CONFIG["LOG_FILE"], 'w')
print(f'Start Time: {date_str}')
