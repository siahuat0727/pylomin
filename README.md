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

## Installation

Add `pylomin` directory to `PYTHONPATH`,

```bash
$ export PYTHONPATH=$PYTHONPATH:/path/to/pylomin
```

and then you can import it like any other Python package.

```python
import pylomin
...
model = pylomin.lazy_loading(model)
...
```

No need other 3rd party Python libraries except `torch`.

## Getting Started

### 1. Lazy-loading

Load model parameters only when needed and release them immediately after use.

```python
model = pylomin.lazy_loading(model)
```

Or provide a list of `target_classes` or `target_modules` to be converted to `lazy-loading` mode.
In addition, when using `target_classes`, you can also provide a list of modules to be skipped.

```python
# Use target_classes
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

## Examples

See `examples/`.
