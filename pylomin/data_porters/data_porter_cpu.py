from .data_porter import DataPorter
from .prefetching_mixin import PrefetchingMixin


class DataPorterCPU(DataPorter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _move_weights(self, module, device):
        for _, param in self.get_direct_parameters(module):
            if param is None:
                continue
            param.data = param.to(device)
        for _, buffer in self.get_direct_buffers(module):
            if buffer is None:
                continue
            buffer.data = buffer.to(device)

    def release_weights(self, module, *_, **_kwargs):
        self._move_weights(module, 'cpu')

    def load_weights(self, module, *_):
        self._move_weights(module, self.computing_device)


# TODO check Tensor.to(non_blocking=True)
class DataPorterCPUPrefetching(PrefetchingMixin, DataPorterCPU):
    pass
