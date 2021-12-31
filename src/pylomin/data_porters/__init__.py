from .data_porter_disk import DataPorterDisk, DataPorterDiskPrefetching
from .data_porter_cpu import DataPorterCPU, DataPorterCPUPrefetching

data_porter_factory = {
    ('disk', False): DataPorterDisk,
    ('disk', True): DataPorterDiskPrefetching,
    ('cpu', False): DataPorterCPU,
    ('cpu', True): DataPorterCPUPrefetching,
}
