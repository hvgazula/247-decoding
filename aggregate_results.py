import glob
import json

import pandas as pd


def get_results():
    results = []
    for filename in glob.glob('results/*/*/*/results.json'):
        with open(filename, 'r') as fp:
            all_results = json.load(fp)

            result = {
                k: v
                for k, v in all_results.items() if k.startswith('avg')
            }
            result.update({
                k: v
                for k, v in all_results['args'].items() if type(v) != list
            })
            results.append(result)

    df = pd.DataFrame(results)
    return df


if __name__ == '__main__':
    df = get_results()
    df.to_csv('results/aggregate.csv', index=False)
