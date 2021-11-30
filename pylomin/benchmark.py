import json
import os
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


def get_inference_latency(forward, warmup_repeat=10, repeat=20, verbose=True):
    def repeat_func(func, repeat, verbose, reduce_func=min):
        runtimes = timeit.repeat(
            func,
            repeat=repeat,
            number=1,
        )
        if verbose:
            print(runtimes)
        return reduce_func(runtimes)

    # warmup
    repeat_func(forward, warmup_repeat, verbose=False)
    return repeat_func(forward, repeat, verbose=verbose)


def evaluate(args, get_model, get_input, apply_optimization):

    model = get_model()
    input_ids = get_input(device=model.device)

    if args.check_equal:
        ground_truth = get_model_forward(model, input_ids)()

    model = apply_optimization(model)
    torch.cuda.reset_peak_memory_stats()

    forward = get_model_forward(model, input_ids)

    if args.check_equal:
        assert all(t1.equal(t2) for t1, t2 in zip(forward(), ground_truth))
        print('check correctness passed!')
        return

    latency = get_inference_latency(forward)
    peak_memory = pytorch_peak_memory_allocated()
    result = {
        'latency': f'{latency:.4f}',
        'memory': f'{peak_memory}',
    }

    Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    device = 'gpu' if args.use_gpu else 'cpu'
    path = os.path.join(
        args.result_dir,
        f'{args.method}@{args.batch_size}@{device}.json'
    )

    with open(path, 'w') as f:
        json.dump(result, f)
    print(args.method, result)
