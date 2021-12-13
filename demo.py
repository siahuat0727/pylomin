from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

import pylomin


def run(args):

    def get_model():
        model = BertModel(BertConfig(
            vocab_size=119547,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
        ))
        model = model.eval()
        if args.use_gpu:
            model = model.cuda()
        return model

    def get_input(vocab_size=119547, device='cpu'):
        input_ids = torch.randint(
            vocab_size,
            (args.batch_size, args.seq_len),
            dtype=torch.long,
            device=device
        )
        return input_ids

    def apply_optimization(model):

        if 'chunked-embedding' in args.method:
            model = pylomin.chunked_embedding(
                model,
                target_module_name='embeddings.word_embeddings',
                # chunk_size=8192,
                chunk_size=4096,
                verbose=True,
            )

        if 'lazy-loading' in args.method:
            # else: Keep a small layer in memory to bypass some troublesome
            # (need to modify huggingface code to fix this)
            skip_modules = [
                model.embeddings.word_embeddings
                if 'keep-embedding' in args.method else
                model.encoder.layer[0].output.LayerNorm
            ]

            model = pylomin.lazy_loading(
                model,
                target_instances=(nn.Linear, nn.Embedding, nn.LayerNorm),
                skip_modules=skip_modules,
                output_dir=args.weight_dir,
                verbose=True,
                prefetch_rule_file=args.prefetch_rule_file,
            )
        return model

    pylomin.evaluate(args, get_model, get_input, apply_optimization,
                     args.warmup_repeat, args.repeat)


def main():

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--method',
                        default='naive',
                        choices=['naive',
                                 'lazy-loading',
                                 'lazy-loading+keep-embedding',
                                 'lazy-loading+prefetching',
                                 'lazy-loading+chunked-embedding'],
                        help='')
    parser.add_argument('--check_equal',
                        action='store_true',
                        help="Whether to check equality")
    parser.add_argument('--use_gpu',
                        action='store_true',
                        help="Whether to use gpu")
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='Batch size')
    parser.add_argument('--seq_len',
                        type=int,
                        default=512,
                        help='Sequence length')
    parser.add_argument('--warmup_repeat',
                        type=int,
                        default=10,
                        help='warmup repeat')
    parser.add_argument('--repeat',
                        type=int,
                        default=50,
                        help='warmup repeat')
    parser.add_argument('--result_dir',
                        default='results',
                        help='Directory to save benchmark results')
    parser.add_argument('--weight_dir',
                        default='weights',
                        help='Directory to save model params '
                             '(for lazy loading)')
    parser.add_argument('--prefetch_rule_file',
                        help='')

    args = parser.parse_args()

    assert ((args.prefetch_rule_file is None and 'prefetching' not in args.method) or
            (args.prefetch_rule_file is not None and 'prefetching' in args.method))

    run(args)


if __name__ == "__main__":
    main()
