from .data_porter_disk import DataPorterDisk, DataPorterDiskPrefetching
from .data_porter_ram import DataPorterRAM, DataPorterRAMPrefetching

data_porter_factory = {
    ('disk', False): DataPorterDisk,
    ('disk', True): DataPorterDiskPrefetching,
    ('RAM', False): DataPorterRAM,
    ('RAM', True): DataPorterRAMPrefetching,
}
