import functools

import torch


# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


def pytorch_memory_allocated():
    return torch.cuda.memory_allocated()//1024//1024


def maybe_print_memory_trace(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        if kwargs.get('verbose', False):
            print(f'Before {func.__name__} {pytorch_memory_allocated()} MB')

        res = func(*args, **kwargs)

        if kwargs.get('verbose', False):
            print(f'After  {func.__name__} {pytorch_memory_allocated()} MB')

        return res

    return wrapper
