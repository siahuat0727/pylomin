import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_result(path):
    method, batch_size, _ = Path(path).stem.split('@')
    with open(path) as f:
        result = json.load(f)
        latency = float(result['latency'])

    gpu_result_path = path.replace('cpu', 'cuda')
    assert Path(gpu_result_path).exists(), (
        'GPU inference is needed for measuring peak memory usage'
    )
    with open(gpu_result_path) as f:
        result_gpu = json.load(f)
        memory = int(result_gpu['memory'])

    return {
        'method': method,
        'batch size': batch_size,
        'throughput (sequences/second)': int(batch_size)/latency,
        'peak memory (MiB)': memory / 1024 / 1024,
    }


def collect_data(paths):
    result_collection = defaultdict(list)

    for path in paths:
        for k, v in get_result(path).items():
            result_collection[k].append(v)

    df = pd.DataFrame.from_dict(result_collection)
    df = df.sort_values(by='batch size', key=pd.to_numeric, kind='stable')
    df = df.sort_values(by='method', key=lambda x: x.str.len(), kind='stable')
    return df


def plot():

    df = collect_data(glob('results/*@*@cpu.json'))
    print(df.to_string(index=False))

    plt.figure(figsize=(10, 6))
    g = sns.relplot(x='batch size', y='peak memory (MiB)', hue='method',
                    size='throughput (sequences/second)', sizes=(100, 800),
                    alpha=0.3, data=df)
    g.fig.suptitle('Peak memory usage versus batch size')
    plt.ylim(0, 2500)

    img_path = 'peak-memory.png'
    print(f'Save {img_path}')
    plt.savefig(img_path)

    plt.figure(figsize=(10, 6))
    g = sns.relplot(x='batch size', y='throughput (sequences/second)',
                    hue='method', size='peak memory (MiB)', sizes=(10, 800),
                    alpha=0.3, data=df)
    g.fig.suptitle('Throughput versus batch size')
    plt.ylim(0, 1.2)

    img_path = 'throughput.png'
    print(f'Save {img_path}')
    plt.savefig(img_path)


if __name__ == "__main__":
    plot()
