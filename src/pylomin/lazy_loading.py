import functools
from itertools import chain

from .data_porters import data_porter_factory


def to_device(module, device, skip_modules=None):
    if skip_modules is None:
        skip_modules = []

    if device == 'cpu':
        return

    def do_to_device(module):
        if module in skip_modules:
            return
        # TODO: Don't access protected attribute
        for _, tensor in chain(module._parameters.items(),
                               module._buffers.items()):
            if tensor is None:
                continue
            tensor.data = tensor.data.to(device)
        for child_module in module.children():
            do_to_device(child_module)

    do_to_device(module)


def get_default_target_modules(model, skip_modules=None):
    if skip_modules is None:
        skip_modules = []

    def has_direct_weights(module):
        # TODO: Don't access protected attribute
        return module._parameters or module._buffers

    return (
        module
        for module in model.modules()
        if (has_direct_weights(module) and
            module not in skip_modules)
    )


def lazy_loading(
    model,
    target_classes=None,
    target_modules=None,
    skip_modules=None,
    device='cpu',
    storage='cpu',
    jit=False,
    load_wrapper=None,
    output_dir='lazy_loading_weights',
    prefetch_rule_file=None,
):
    r"""jit is needed only to bypass torch.jit.trace bug"""
    if skip_modules is None:
        skip_modules = []

    def get_data_porter():
        do_prefetch = prefetch_rule_file is not None
        data_porter_cls = data_porter_factory.get(
            (storage, do_prefetch))
        assert data_porter_cls is not None, (
            f'Not support storage={storage}'
            f'prefetching={do_prefetch}'
        )
        return data_porter_cls(model,
                               computing_device=device,
                               prefetch_rule_file=prefetch_rule_file,
                               weight_dir=output_dir)

    if target_classes is not None:
        assert target_modules is None, (
            'Can\'t accept both target_classes and target_modules'
        )
        skip_modules = set(skip_modules)
        target_modules = (
            module
            for module in model.modules()
            if (isinstance(module, target_classes)
                and module not in skip_modules)
        )
    if target_modules is None:
        target_modules = get_default_target_modules(model,
                                                    skip_modules=skip_modules)

    target_modules = set(target_modules)

    data_porter = get_data_porter()
    load_weights_hook = data_porter.load_weights
    if load_wrapper is not None:
        load_weights_hook = load_wrapper(load_weights_hook)

    for module in target_modules:
        data_porter.release_weights(module, first_time=True)

        if jit:
            # https://github.com/pytorch/pytorch/issues/70511
            # need to do more than once to work properly
            # TODO dig into torch.jit.trace code to learn more
            module.register_forward_pre_hook(load_weights_hook)
            module.register_forward_hook(data_porter.release_weights)
        module.register_forward_pre_hook(load_weights_hook)
        module.register_forward_hook(data_porter.release_weights)

    # For those with parameters but not in target_modules,
    # move to compute device
    to_device(model, device, skip_modules=target_modules)

    return model
