from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pylomin
import torch
import transformers


def main(args):
    input_ids = torch.randint(50257, (1, 64), dtype=torch.long, device='cuda')
    model = transformers.GPT2Model.from_pretrained('gpt2-large').eval()

    if 'chunked-embedding' in args.method:
        model = pylomin.chunked_embedding(model, 'wte', chunk_size=4096)
        model = pylomin.chunked_embedding(model, 'wpe', chunk_size=4096)
    if 'lazy-loading' in args.method:
        model = pylomin.lazy_loading(model, device='cuda', storage_device='RAM')
    else:
        model = model.cuda()

    with torch.no_grad():
        outputs = model(input_ids)

    peak_memory = torch.cuda.memory_stats()['allocated_bytes.all.peak']
    print(f'Peak memory: {peak_memory/(2**20):.2f} MiB')


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--method',
                        nargs='+',
                        default=['naive'],
                        choices=['naive',
                                 'lazy-loading',
                                 'chunked-embedding'],
                        help='')
    args = parser.parse_args()
    main(args)
