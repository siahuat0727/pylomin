# BERT demo

We provide a script to measure the peak memory usage and throughput of a BERT-like model inference with different optimization approaches.

In summary, we can reduce the peak memory by **36×** while maintaining 66% model throughput in this scenario.

## Execution

Before you start, you may need to install the dependencies of this demo:

```bash
python3 -m pip install requirements.txt
```

Run script:

```bash
bash run.sh
```

## Description of different methods

+ **naive**
  + Naïve PyTorch inference
  + peak memory **1672 MiB**, throughput 0.96 (batch size=1)
+ **lazy-loading**
  + Load model parameters only when needed and release them immediately after use
  + peak memory **469 MiB**, throughput 0.60 (batch size=1)
+ **keep-embedding**
  + Keep `embeddings.word_embeddings` layer persistently in memory to increase throughput
  + peak memory **513 MiB**, throughput 0.71 (batch size=1)
+ **chunked-embedding**
  + Split `embeddings.word_embeddings` layer into smaller chunks to reduce peak memory
  + peak memory **46 MiB**, throughput 0.64 (batch size=1)

> Use `torch.cuda.memory_stats()['allocated_bytes.all.peak']` to get peak memory

## More results

```
                         method batch size  throughput (sequences/second)  peak memory (MiB)
                          naive          1                       0.963855        1672.234375
                          naive          2                       0.956572        1718.246094
                          naive          4                       0.953584        1810.269531
                          naive          8                       1.012927        1994.316406
                          naive         16                       0.990712        2362.410156
                   lazy-loading          1                       0.604924         469.003906
                   lazy-loading          2                       0.707014         471.015625
                   lazy-loading          4                       0.815494         475.039062
                   lazy-loading          8                       0.913816         483.085938
                   lazy-loading         16                       0.941055         736.199219
    lazy-loading+keep-embedding          1                       0.711541         512.996094
    lazy-loading+keep-embedding          2                       0.786782         559.007812
    lazy-loading+keep-embedding          4                       0.849005         651.031250
    lazy-loading+keep-embedding          8                       0.956961         835.078125
    lazy-loading+keep-embedding         16                       0.953550        1203.171875
 lazy-loading+chunked-embedding          1                       0.636618          46.023438
 lazy-loading+chunked-embedding          2                       0.762806          92.035156
 lazy-loading+chunked-embedding          4                       0.828878         184.058594
 lazy-loading+chunked-embedding          8                       0.930233         368.105469
 lazy-loading+chunked-embedding         16                       0.946942         736.199219
```

```bash
python3 plot.py
```

| Peak memory vs. batch size | Throughput vs. batch size |
| -------- | -------- |
|  [![](https://i.imgur.com/F52U6XL.png)](https://i.imgur.com/F52U6XL.pn)   | [![](https://i.imgur.com/YQu0k3J.png)](https://i.imgur.com/YQu0k3J.png)  |


The environment used to run the above benchmarks:

```
$ lscpu
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              6
On-line CPU(s) list: 0-5
Thread(s) per core:  1
Core(s) per socket:  6
Socket(s):           1
NUMA node(s):        1
Vendor ID:           GenuineIntel
CPU family:          6
Model:               79
Model name:          Intel(R) Xeon(R) CPU E5-2690 v4 @ 2.60GHz
Stepping:            1
CPU MHz:             2593.992
BogoMIPS:            5187.98
Hypervisor vendor:   Microsoft
Virtualization type: full
L1d cache:           32K
L1i cache:           32K
L2 cache:            256K
L3 cache:            35840K
NUMA node0 CPU(s):   0-5
```
