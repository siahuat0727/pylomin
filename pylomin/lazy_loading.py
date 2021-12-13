import functools
import json
import os
import queue
import threading
import time
from functools import partial
from itertools import chain
from pathlib import Path

import torch

from .utils import maybe_print_gpu_memory_trace


@maybe_print_gpu_memory_trace
def lazy_loading(
    model,
    target_instances=None,
    target_modules=None,
    skip_modules=[],
    output_dir='lazy_loading_weights',
    prefetch_rule_file=None,
    verbose=False,
):
    do_prefetch = prefetch_rule_file is not None

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
        data = torch.load(module.param_path)
        for name, param in data['parameters']:
            module.register_parameter(name, param)
        for name, buffer in data['buffers']:
            module.register_buffer(name, buffer)
        module.loaded = True

    def module_wait_weights(module, *_):
        # if module.loaded:
        #     print('Success prefetch! no need to wait')
        if not module.loading:
            print(f'Warning, not loading {module.param_path}')
            module_load_weights(module)
            return
        while not module.loaded:
            assert module.loading
            time.sleep(0.000001)
        # print('Not prefetch! wait some time')

    def lazy_loading_wrapper(func, module):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            tic = time.time()
            if do_prefetch:
                targets = prefetch_rules.get(module, [])
                for tgt_module in targets:
                    tgt_module.loading = True
                    model.queue.put(partial(module_load_weights, tgt_module))
                module_wait_weights(module)
            else:
                module_load_weights(module)
            toc = time.time()
            module.load_time = toc - tic

            tic = time.time()
            res = func(*args, **kwargs)
            module.forward_time = time.time() - tic

            module_delete_weights(module)
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
    target_modules = set(target_modules)

    if do_prefetch:

        name2module = {
            name: module
            for name, module in model.named_modules()
        }
        with open(prefetch_rule_file) as f:
            prefetch_rules = {
                name2module[key]: [name2module[v] for v in value]
                for key, value in json.load(f).items()
            }

        def worker():
            while True:
                job = model.queue.get()
                job()
                model.queue.task_done()

        model.queue = queue.Queue()
        threading.Thread(target=worker, daemon=True).start()

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for name, module in model.named_modules():
        if module not in target_modules:
            continue
        module.param_path = os.path.join(output_dir, f'{name}.pt')

        module_save_weights(module)
        module_delete_weights(module)

        module.forward = lazy_loading_wrapper(module.forward, module)
        # module.register_forward_pre_hook(module_load_weights)
        # module.register_forward_hook(module_delete_weights)

    return model
