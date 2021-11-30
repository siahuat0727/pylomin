# Pylomin 

Pylomin (**PY**torch **LO**w-**M**emory **IN**ference) is a library for low-memory inferencing in PyTorch.

## Installation

...

## Usage

For example, the following code snippet converts all `nn.Linear` and `nn.Embedding` modules in the model into `lazy-loading` mode.

```python
model = pylomin.lazy_loading(model, target_instances=(nn.Linear, nn.Embedding))
```

Or, provide a list of modules via `target_modules`.

```python
model = pylomin.lazy_loading(model, target_modules=your_target_modules)
```

A detailed documentation is undergoing! :)

## Methods

### 1. Lazy-loading
...

### 2. Grouped-embedding
...

### 3. Prefetching
...

## Results

```bash
bash run.sh
python plot.py
```

