# GPT-2 inference demo

```
$ python3 examples/gpt2/demo.py --method naive
Peak memory: 3101.94 MiB

$ python3 examples/gpt2/demo.py --method lazy-loading
Peak memory: 246.31 MiB

$ python3 examples/gpt2/demo.py --method lazy-loading chunked-embedding
Peak memory: 63.19 MiB
```
