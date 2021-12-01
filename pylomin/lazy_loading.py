import functools
import os
from itertools import chain
from pathlib import Path

import torch

from .utils import maybe_print_gpu_memory_trace


@maybe_print_gpu_memory_trace
def lazy_loading(
    model,
    target_instances=None,
    target_modules=None,
    skip_modules=None,
    output_dir='lazy_loading_weights',
    verbose=False,
):

    def module_save_weights(module, path):
        torch.save({
            'parameters': list(module.named_parameters()),
            'buffers': list(module.named_buffers()),
        }, path)

    def module_delete_weights(module):
        weight_names = [
            name
            for name, _ in chain(
                module.named_parameters(), module.named_buffers())
        ]
        for name in weight_names:
            setattr(module, name, None)

    def module_load_weights(module, path):
        data = torch.load(path)
        for name, param in data['parameters']:
            module.register_parameter(name, param)
        for name, buffer in data['buffers']:
            module.register_buffer(name, buffer)

    def lazy_loading_wrapper(func, module, path):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            module_load_weights(module, path)
            res = func(*args, **kwargs)
            module_delete_weights(module)
            return res
        return wrapper

    assert target_instances is not None or target_modules is not None, (
        'Must provide either one'
    )
    assert not (target_instances is not None and target_modules is not None), (
        'Can\'t provide both'
    )

    # if target_modules is None:
    #     target_modules = [
    #             module
    #             for module in
    #     ]

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for name, module in model.named_modules():
        if name in skip_modules:
            continue
        path = os.path.join(output_dir, f'{name}.pt')
        if isinstance(module, target_instances):
            module_save_weights(module, path)
            module_delete_weights(module)
            module.forward = lazy_loading_wrapper(module.forward, module, path)

    return model
