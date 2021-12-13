import copy
import functools
import json
import os
from collections import defaultdict
from itertools import accumulate, chain
from pathlib import Path

import torch


def get_model_forward(model, input_ids):
    def forward():
        with torch.no_grad():
            return model(input_ids)
    return forward


def mem_analysis(
    model,
    target_instances=None,
    target_modules=None,
    skip_modules=[],
    output_dir='lazy_loading_weights',
):

    def module_save_weights(module):
        torch.save({
            'parameters': list(module.named_parameters()),
            'buffers': list(module.named_buffers()),
        }, module.param_path)

    def module_delete_weights(module, *_):
        weight_names = [
            name
            for name, _ in chain(
                module.named_parameters(), module.named_buffers())
        ]
        for name in weight_names:
            setattr(module, name, None)
        module.loaded = False
        module.loading = False

    def module_load_weights(module, *_):
        r""" Load weights and return memory need for weights """
        def do_load_weights():
            data = torch.load(module.param_path)
            for name, param in data['parameters']:
                module.register_parameter(name, param)
            for name, buffer in data['buffers']:
                module.register_buffer(name, buffer)

        mem_b4_load = torch.cuda.memory_allocated()
        do_load_weights()
        module.param_memory = torch.cuda.memory_allocated() - mem_b4_load

    def mem_analysis_wrapper(func, module):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            torch.cuda.reset_peak_memory_stats()
            model.module_order.append(module)

            module_load_weights(module)
            res = func(*args, **kwargs)
            module_delete_weights(module)

            module.peak_memory = torch.cuda.memory_stats()[
                'allocated_bytes.all.peak']
            return res
        return wrapper

    assert target_instances is not None or target_modules is not None, (
        'Must provide either one'
    )
    assert not (target_instances is not None and target_modules is not None), (
        'Can\'t provide both'
    )

    if target_modules is None:
        skip_modules = set(skip_modules)
        target_modules = (
            module
            for module in model.modules()
            if (isinstance(module, target_instances)
                and module not in skip_modules)
        )

    model.module_order = []

    target_modules = set(target_modules)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for name, module in model.named_modules():
        if module not in target_modules:
            continue
        module.param_path = os.path.join(output_dir, f'{name}.pt')

        module_save_weights(module)
        module_delete_weights(module)

        module.forward = mem_analysis_wrapper(module.forward, module)
        # module.register_forward_pre_hook(module_load_weights)
        # module.register_forward_hook(module_delete_weights)

    return model


def generate_prefetching_rule(model, input_ids, target_modules,
                              target_peak_memory=None,
                              file_path='prefetch_rule.json'):
    model = mem_analysis(model, target_modules=target_modules)

    # dummy inference
    get_model_forward(model, input_ids)()

    assert len(target_modules) == len(model.module_order)

    peak_memory_list = [module.peak_memory for module in model.module_order]

    if target_peak_memory is None:
        target_peak_memory = max(peak_memory_list)
    else:
        # TODO print some info
        target_peak_memory = max(target_peak_memory, max(peak_memory_list))
    print(f'peak {target_peak_memory}')

    param_memory_list = [module.param_memory for module in model.module_order]
    param_mem_prefixsum = list(accumulate(param_memory_list+param_memory_list))
    n_target = len(model.module_order)

    def param_mem_range_sum(i, j):
        """ Get total param mem of modules in range [i, j] """
        assert i <= j, (i, j)
        return param_mem_prefixsum[j] - param_mem_prefixsum[i-1]

    module2name = {
        module: name
        for name, module in model.named_modules()
    }

    prefetch_rule = defaultdict(list)
    target_prefetch_i = 0

    prefetch_rules = []
    counter = defaultdict(int)
    for module_i, module in enumerate(model.module_order):
        while target_prefetch_i < 2*len(target_modules) and all(
                peak_memory_list[i % n_target] + param_mem_range_sum(
                    i+1, target_prefetch_i) <= target_peak_memory
                for i in range(module_i, target_prefetch_i)
        ):
            prefetched_module = model.module_order[target_prefetch_i % n_target]
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
