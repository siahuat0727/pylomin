import json
import os
import timeit
from pathlib import Path

import torch
import torch.nn as nn

from .generate_prefetching_rule import generate_prefetching_rule


def pytorch_peak_memory_allocated():
    return torch.cuda.memory_stats()['allocated_bytes.all.peak']


def get_model_forward(model, input_ids):
    def forward():
        with torch.no_grad():
            return model(input_ids)
    return forward


def get_inference_latency(forward, warmup_repeat=10, repeat=50, verbose=True):
    def repeat_func(func, repeat, verbose, reduce_func=min):
        runtimes = timeit.repeat(
            func,
            repeat=repeat,
            number=1,
        )
        if verbose:
            print(runtimes)
        return reduce_func(runtimes)

    if warmup_repeat > 0:
        print('Warmup... ')
        repeat_func(forward, warmup_repeat, verbose=False)
    print('Testing latency...')
    return repeat_func(forward, repeat, verbose=verbose)


def evaluate(args, get_model, get_input, apply_optimization, warmup_repeat=10, repeat=50):

    model = get_model()
    input_ids = get_input(device=model.device)

    if args.check_equal:
        ground_truth = get_model_forward(model, input_ids)()

    skip_modules = [
        model.embeddings.word_embeddings
        if 'keep-embedding' in args.method else
        model.encoder.layer[0].output.LayerNorm
    ]

    target_instances = (nn.Linear, nn.Embedding, nn.LayerNorm)
    target_modules = [
        module
        for module in model.modules()
        if (isinstance(module, target_instances)
            and module not in skip_modules)
    ]

    if (args.prefetch_rule_file is not None
            and not os.path.isfile(args.prefetch_rule_file)):
        generate_prefetching_rule(model, input_ids, target_modules,
                                  file_path=args.prefetch_rule_file)
        # TODO lazily move to gpu so that no need to exit and run again
        print(f'{args.prefetch_rule_file} created, please run again')
        return

    model = apply_optimization(model)

    forward = get_model_forward(model, input_ids)

    if args.check_equal:
        assert all(t1.equal(t2) for t1, t2 in zip(forward(), ground_truth))
        print('check correctness passed!')
        return

    torch.cuda.reset_peak_memory_stats()

    latency = get_inference_latency(forward, warmup_repeat, repeat)

    print('load time', sum(
        module.load_time
        for module in target_modules
    ))
    print('forward time', sum(
        module.forward_time
        for module in target_modules
    ))

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
