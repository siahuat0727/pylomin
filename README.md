# Pylomin

Pylomin (**PY**torch **LO**w-**M**emory **IN**ference) is a deep learning optimization library for low-memory inferencing in PyTorch.

## Motivation

The scale of deep learning models has grown exponentially in recent years, which has greatly increased the difficulty of product deployment.

![](https://www.microsoft.com/en-us/research/uploads/prod/2021/10/model-size-graph.jpg)

 <p align = "center">
Image source: <a href="https://www.microsoft.com/en-us/research/blog/using-deepspeed-and-megatron-to-train-megatron-turing-nlg-530b-the-worlds-largest-and-most-powerful-generative-language-model/">Microsoft Research Blog</a>
</p>


The goal of this library is to enable low-cost deployment of deep learning models:

+ Extremely low memory requirement
  + For example, we can reduce the peak memory requirement for the inference of a BERT-like model (with 1.6 GiB parameters) to 46 MiB.
+ Minimize memory requirements while maintaining the model throughput
  + Eliminate the time waiting for parameters to load by prefetching (under development)
  + > TODO: add a number here after development

> Peak memory is the maximum amount of memory needed to store model parameters and hidden states at any time during the model inference.


## Getting Started

### 1. Lazy-loading

Load model parameters only when needed and delete them immediately after use.

Provide a list of `target_classes` or `target_modules` to be converted to `lazy-loading` mode.
In addition, when using `target_classes`, you can also provide a list of modules to be skipped.

```python
# Use target_classes
model = pylomin.lazy_loading(model, target_classes=[nn.Linear, nn.Embedding])
model = pylomin.lazy_loading(model, target_classes=[nn.Linear, nn.Embedding],
                             skip_modules=[model.embeddings.word_embeddings])

# Use target_modules
target_modules = [module for module in model.modules() if some_condition]
model = pylomin.lazy_loading(model, target_modules=target_modules)
```

### 2. Chunked-embedding

Attempts to split an `torch.nn.Embedding` layer into multiple chunks with each has `num_embeddings` equal to `chunk_size`, except the last one.

```python
model = pylomin.chunked_embedding(model,
                                  target_module_name='embeddings.word_embeddings',
                                  chunk_size=2048)
```

## Demo

We provide a script to measure the peak memory usage and throughput of a model inference with different optimization approaches.

In summary, we can reduce the peak memory by **36×** while maintaining 66% model throughput in our scenario.

### Execution

```bash
bash run.sh
```

### Methods

+ naive
  + Naïve PyTorch inference
+ lazy-loading
  + Load model parameters only when needed and delete them immediately after use
+ keep-embedding
  + Keep `embeddings.word_embeddings` layer persistently in memory to increase throughput
+ chunked-embedding
  + Split `embeddings.word_embeddings` layer into smaller chunks to reduce peak memory usage (when using lazy-loading)

### Results

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
