import functools
import json
import os
import time
import timeit
from pathlib import Path

import torch


def pytorch_peak_memory_allocated():
    return torch.cuda.memory_stats()['allocated_bytes.all.peak']


def get_model_forward(model, input_ids):

    def forward():
        with torch.no_grad():
            return model(input_ids)
    return forward


def get_latency(forward, warmup_repeat=10, repeat=50, verbose=True,
                reduce_func=min):
    def repeat_func(func, repeat, verbose, reduce_func=reduce_func):
        runtimes = timeit.repeat(
            func,
            repeat=repeat,
            number=1,
        )
        if verbose:
            print(runtimes)
        return reduce_func(runtimes)

    if repeat <= 0:
        raise AssertionError

    if warmup_repeat > 0:
        print('Warmup... ')
        repeat_func(forward, warmup_repeat, verbose=False)
    print('Testing latency...')
    return repeat_func(forward, repeat, verbose=verbose)


def record_load_time_wrapper(func):
    @functools.wraps(func)
    def wrapper(module, *args, **kwargs):
        tic = time.time()
        res = func(module, *args, **kwargs)
        toc = time.time()
        module.load_time = toc - tic
        return res
    return wrapper


def evaluate(args, get_model, get_input_ids, apply_optimization,
             warmup_repeat=10, repeat=50, verbose=True):

    input_ids = get_input_ids().to(args.device)
    model = get_model()
    model = apply_optimization(model, input_ids)

    torch.cuda.reset_peak_memory_stats()

    forward = get_model_forward(model, input_ids)
    latency = get_latency(forward, warmup_repeat, repeat, verbose)
    peak_memory = pytorch_peak_memory_allocated()

    result = {
        'latency': f'{latency:.4f}',
        'batch-size': args.batch_size,
        'throughput': f'{args.batch_size/latency:.2f}',
        'memory': peak_memory,
    }

    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    path = os.path.join(
        args.result_dir,
        f'{args.method}@{args.batch_size}@{args.device}.json'
    )

    with open(path, 'w') as f:
        json.dump(result, f)
    print(args.method, f'{args.device}+{args.storage}', result)
    print(f'Peak memory: {peak_memory/1024/1024:.2f} MiB')
    print(f'Throughput: {args.batch_size/latency:.3f} seq/s')
    print()
