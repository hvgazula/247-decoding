import argparse

import matplotlib.pyplot as plt

from aggregate_results import get_results

parser = argparse.ArgumentParser()
parser.add_argument("-q", nargs="+", required=True)
parser.add_argument("-x", type=str, required=True)
parser.add_argument("-y", type=str, required=True)

args = parser.parse_args()
print(args)

# TODO - choose one.
df = get_results()
# df = pd.read_csv('results/aggregate.csv')

for query in args.q:
    sub_df = df.query(query).sort_values(by=args.x)
    x = sub_df[args.x] / 512  # NOTE - hardcoded.
    y = sub_df[args.y]
    plt.plot(x, y, marker='.')

plt.xlabel(args.x)
plt.ylabel(args.y)
plt.grid()
plt.savefig('results/plots/out.png')
