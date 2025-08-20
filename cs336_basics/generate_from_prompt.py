import argparse
from argparse import ArgumentParser
import numpy as np
import numpy.typing as npt
import os
from pathlib import Path
import time
import torch

from cs336_basics.args import get_parser
import cs336_basics.run_tokenizer as run_tokenizer
from cs336_basics.run_tokenizer import BpeTokenizer
from cs336_basics.model.model import TransformerLm
from cs336_basics.model.data import get_batch
from cs336_basics.model.nn_utils import cross_entropy_loss
from cs336_basics.model.serialization import load_checkpoint


def parse_args_extra():
    parser = get_parser("generate from prompt")

    parser.add_argument(
        '--model-checkpoint',
        type=str,
        default='data/model_tiny_stories/1st/model-checkpoint-step-4882',
    )

    parser.add_argument(
        '--vocab-path',
        type=str,
        default=run_tokenizer.TINY_STORIES_TOKENIZER_VOCAB_PATH,
    )

    parser.add_argument(
        '--merges-path',
        type=str,
        default=run_tokenizer.TINY_STORIES_TOKENIZER_MERGES_PATH,
    )

    parser.add_argument(
        '-p',
        '--prompt',
        type=str,
        default='Just a short story: ',
    )

    parser.add_argument(
        '-n',
        '--max-response',
        type=int,
        default='200',
    )

    parser.add_argument(
        '-t',
        '--generate-temperature',
        type=float,
        default=1.0,
    )

    parser.add_argument(
        '-k',
        '--generate-top-k',
        type=int,
        default=None,
    )

    return  parser.parse_args()


def get_tokenizer(args: ArgumentParser):
    return BpeTokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens=['<|endoftext|>'])


def get_transformer_lm(args: ArgumentParser):
    model = TransformerLm(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )

    load_checkpoint(
        src=Path(args.root_folder) / args.model_checkpoint,
        model=model,
        optimizer=None,
    )

    return model


if __name__ == '__main__':
    args = parse_args_extra()

    tokenizer = get_tokenizer(args)
    model = get_transformer_lm(args)

    if args.prompt == '':
        args.prompt = input('Your prompt: ')

    input_tokens = tokenizer.encode(args.prompt)
    eos_token_id = tokenizer.encode('<|endoftext|>')[0]

    print(input_tokens)
    
    resp = model.generate(
        text=torch.tensor(input_tokens, device=args.device),
        max_response=args.max_response,
        eos_token_id=eos_token_id,
        temperature=args.generate_temperature,
        top_k=args.generate_top_k)

    print(f'generated {resp.shape[-1]} tokens.\n')
    print('-' * 40 + ' prompt ' + '-' * 40)
    print(args.prompt)
    print('+' * 88)
    print(tokenizer.decode(resp.tolist()))
    print('-' * 88)
