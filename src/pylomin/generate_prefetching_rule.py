import copy
import functools
import json
from collections import defaultdict
from itertools import accumulate

import torch

from .lazy_loading import lazy_loading


def get_model_forward(model, input_ids):
    def forward():
        with torch.no_grad():
            return model(input_ids)
    return forward


def record_peak_memory(target_modules):
    def record_peak_memory_wrapper(func, module):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            torch.cuda.reset_peak_memory_stats()
            res = func(*args, **kwargs)
            module.peak_memory = torch.cuda.memory_stats()[
                'allocated_bytes.all.peak']
            return res
        return wrapper

    for module in target_modules:
        module.forward = record_peak_memory_wrapper(module.forward, module)


def record_module_order(target_modules):
    module_order = []

    def record_peak_memory_wrapper(func, module):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            res = func(*args, **kwargs)
            module_order.append(module)
            return res
        return wrapper

    for module in target_modules:
        module.forward = record_peak_memory_wrapper(module.forward, module)
    return module_order


def record_load_mem_wrapper(func):
    @functools.wraps(func)
    def wrapper(module, *args, **kwargs):
        lo = torch.cuda.memory_allocated()
        res = func(module, *args, **kwargs)
        hi = torch.cuda.memory_allocated()
        module.param_memory = hi - lo
        return res
    return wrapper


def generate_prefetching_rule(model, input_ids, target_modules,
                              target_peak_memory=None,
                              file_path='prefetch_rule.json'):
    target_module_names = {
        name
        for name, module in model.named_modules()
        if module in target_modules
    }
    model = copy.deepcopy(model)
    target_modules = [
        module
        for name, module in model.named_modules()
        if name in target_module_names
    ]

    model = lazy_loading(model,
                         target_modules=target_modules,
                         device='cuda',
                         storage='cpu',
                         load_wrapper=record_load_mem_wrapper)
    record_peak_memory(target_modules)
    module_order = record_module_order(target_modules)

    # dummy inference
    get_model_forward(model, input_ids)()

    if len(target_modules) != len(module_order):
        raise AssertionError(
            (len(target_modules), len(module_order)
        ))

    peak_memory_list = [module.peak_memory for module in module_order]

    if target_peak_memory is None:
        target_peak_memory = max(peak_memory_list)
    else:
        # TODO print some info
        target_peak_memory = max(target_peak_memory, max(peak_memory_list))
    print(f'peak {target_peak_memory}')

    param_memory_list = [module.param_memory for module in module_order]
    param_mem_prefixsum = list(accumulate(param_memory_list+param_memory_list))
    n_target = len(module_order)

    def param_mem_range_sum(i, j):
        """ Get total param mem of modules in range [i, j] """
        if i > j:
            raise AssertionError(i, j)
        return param_mem_prefixsum[j] - param_mem_prefixsum[i-1]

    module2name = {
        module: name
        for name, module in model.named_modules()
    }

    prefetch_rule = defaultdict(list)
    target_prefetch_i = 0

    prefetch_rules = []
    counter = defaultdict(int)
    for module_i, module in enumerate(module_order):
        while target_prefetch_i < 2*len(target_modules) and all(
                peak_memory_list[i % n_target] + param_mem_range_sum(
                    i+1, target_prefetch_i) <= target_peak_memory
                for i in range(module_i, target_prefetch_i)
        ):
            prefetched_module = module_order[target_prefetch_i % n_target]
            counter[prefetched_module] += 1

            prefetch_rules.append((module, prefetched_module))

            print(f'{module_i} prefetch {target_prefetch_i % n_target} '
                  f'{module2name[module]} -> {module2name[prefetched_module]}')
            target_prefetch_i += 1

    print('---')
    for module, prefetched_module in prefetch_rules:
        if counter[prefetched_module] > 1:
            counter[prefetched_module] -= 1
            continue
        prefetch_rule[module2name[module]].append(
            module2name[prefetched_module])
        print(f'{module2name[module]} -> {module2name[prefetched_module]}')

    with open(file_path, 'w') as f:
        json.dump(prefetch_rule, f)
