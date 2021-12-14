import functools
import time
from itertools import chain
from pathlib import Path

from .data_porters import data_porter_factory
from .utils import maybe_print_gpu_memory_trace


def to_device(module, device, skip_modules=[]):

    if device == 'cpu':
        return

    def do_to_device(module):
        if module in skip_modules:
            return
        # TODO: Don't access protected attribute
        for name, tensor in chain(module._parameters.items(),
                                  module._buffers.items()):
            tensor.data = tensor.data.to(device)
        for child_module in module.children():
            do_to_device(child_module)

    do_to_device(module)


@maybe_print_gpu_memory_trace
def lazy_loading(
    model,
    target_instances=None,
    target_modules=None,
    skip_modules=[],
    output_dir='lazy_loading_weights',
    prefetch_rule_file=None,
    device='cpu',
    storage_device='disk',
    load_wrapper=None,
    verbose=False,
):

    def get_data_porter():
        do_prefetch = prefetch_rule_file is not None
        data_porter_cls = data_porter_factory.get(
            (storage_device, do_prefetch))
        assert data_porter_cls is not None, (
            f'Not support {storage_device=}'
            f'prefetching={do_prefetch}'
        )
        return data_porter_cls(model,
                               computing_device=device,
                               prefetch_rule_file=prefetch_rule_file)

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

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    data_porter = get_data_porter()
    load_weights_hook = data_porter.load_weights
    if load_wrapper is not None:
        load_weights_hook = load_wrapper(load_weights_hook)

    for module in target_modules:
        data_porter.release_weights(module, first_time=True)

        module.register_forward_pre_hook(load_weights_hook)
        module.register_forward_hook(data_porter.release_weights)

    # For those with parameters but not in target_modules,
    # move to compute device
    to_device(model, device, skip_modules=target_modules)

    return model
