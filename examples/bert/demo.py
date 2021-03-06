import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import pylomin
import torch
import torch.nn as nn
from pylomin.benchmark import evaluate
from transformers import BertConfig, BertModel


def run(args):

    def apply_optimization(model, input_ids):

        if 'chunked-embedding' in args.method:
            model = pylomin.chunked_embedding(
                model,
                target_module_name='embeddings.word_embeddings',
                chunk_size=4096,
            )

        if 'lazy-loading' in args.method:

            target_classes = (nn.Linear, nn.Embedding, nn.LayerNorm),

            # else: Keep a small layer in memory to bypass some troublesome
            # (need to modify huggingface code to fix this)
            skip_modules = [
                model.embeddings.word_embeddings
                if 'keep-embedding' in args.method else
                model.encoder.layer[0].output.LayerNorm
            ]

            target_modules = [
                module
                for module in model.modules()
                if (isinstance(module, target_classes)
                    and module not in skip_modules)
            ]

            if (args.prefetch_rule_file is not None and
                    not os.path.isfile(args.prefetch_rule_file)):
                pylomin.generate_prefetching_rule(
                    model, input_ids, target_modules,
                    file_path=args.prefetch_rule_file)

            model = pylomin.lazy_loading(
                model,
                target_modules=target_modules,
                output_dir=args.weight_dir,
                prefetch_rule_file=args.prefetch_rule_file,
                device=args.device,
                storage=args.storage,
            )
        else:
            model.to(args.device)
        return model

    vocab_size = 119547

    def get_model():
        model = BertModel(BertConfig(
            vocab_size=vocab_size,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
        )).eval()
        return model

    def get_input_ids():
        input_ids = torch.randint(
            vocab_size,
            (args.batch_size, args.seq_len),
            dtype=torch.long,
        )
        return input_ids

    evaluate(args, get_model, get_input_ids, apply_optimization,
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
    parser.add_argument('--device',
                        default='cpu',
                        help="Computing device")
    parser.add_argument('--storage',
                        default='disk',
                        help='Storage device')
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

    if 'prefetching' in args.method:
        if args.prefetch_rule_file is None:
            raise AssertionError(
                'Please provide prefetch_rule_file'
            )

    run(args)


if __name__ == "__main__":
    main()
