from itertools import chain

import torch

from .data_porter import DataPorter
from .prefetching_mixin import PrefetchingMixin


class DataPorterDisk(DataPorter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save_weights(self, module):
        module = module.to(self.computing_device)
        torch.save({
            'parameters': list(module.named_parameters()),
            'buffers': list(module.named_buffers()),
        }, self.get_save_path(module))

    def release_weights(self, module, *_, first_time=False):
        if first_time:
            self._save_weights(module)

        weight_names = [
            name
            for name, _ in chain(
                module.named_parameters(), module.named_buffers())
        ]
        for name in weight_names:
            setattr(module, name, None)
        module.loaded = False
        module.loading = False

    def load_weights(self, module, *_):
        data = torch.load(self.get_save_path(module))
        for name, param in data['parameters']:
            module.register_parameter(name, param)
        for name, buffer in data['buffers']:
            module.register_buffer(name, buffer)


class DataPorterDiskPrefetching(PrefetchingMixin, DataPorterDisk):
    pass
