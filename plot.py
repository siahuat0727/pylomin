from collections import Counter, defaultdict
from glob import glob

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_result(path):
    with open(path.replace('#', 'latency')) as f:
        latency = float(f.readline().strip())
    with open(path.replace('#', 'memory')) as f:
        memory = float(f.readline().strip())
    method, _, batch_size = path.split('/')[-1].split('@')
    return {
        'method': method,
        'batch_size': batch_size,
        'peak-memory (MB)': memory,
        'throughput (sequences/second)': int(batch_size)/latency,
    }


def get_input(paths):
    def encode(path):
        return path.replace('latency', '#').replace('memory', '#')

    counter = Counter([
        encode(path)
        for path in paths
    ])
    return [k for k, v in counter.items() if v == 2]


def main():
    paths = glob('results/*@*@*')
    result_collection = defaultdict(list)

    for target in get_input(paths):
        for k, v in get_result(target).items():
            result_collection[k].append(v)

    print(result_collection)

    df = pd.DataFrame.from_dict(result_collection)

    df = df.sort_values(by='batch_size', key=pd.to_numeric, kind='stable')
    df = df.sort_values(by='method', key=lambda x: x.str.len(), kind='stable')
    print(df.to_string(index=False))

    plt.figure(figsize=(10, 6))
    g = sns.relplot(x='batch_size', y='peak-memory (MB)', hue='method',
                    size='throughput (sequences/second)', sizes=(100, 400),
                    alpha=0.3, data=df)

    img_path = f'memory.png'
    print(f'Save {img_path}')
    plt.savefig(img_path)

    plt.figure(figsize=(10, 6))
    g = sns.relplot(x='batch_size', y='throughput (sequences/second)', hue='method',
                    size='peak-memory (MB)', sizes=(5, 800),
                    alpha=0.3, data=df)
    plt.ylim(0, 1.2)

    img_path = f'throughput.png'
    print(f'Save {img_path}')
    plt.savefig(img_path)


main()
