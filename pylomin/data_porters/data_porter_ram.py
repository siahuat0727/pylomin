from .data_porter import DataPorter
from .prefetching_mixin import PrefetchingMixin


class DataPorterRAM(DataPorter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def release_weights(self, module, *_, **kwargs):
        module.to('cpu')

    def load_weights(self, module, *_):
        module.to(self.computing_device)


# TODO check Tensor.to(non_blocking=True)
class DataPorterRAMPrefetching(PrefetchingMixin, DataPorterRAM):
    pass
