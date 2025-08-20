import argparse
import numpy as np
import torch

from cs336_basics.model.data import get_batch
from cs336_basics.model.model import TransformerLm
from cs336_basics.model.nn_utils import cross_entropy_loss
from cs336_basics.model.optimizer import AdamW
from cs336_basics.model.serialization import load_checkpoint, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="basic model trainier")

    parser.add_argument(
        '--model-folder',
        type=str,
        default='data/model_tiny_stories/1st',
    )

    parser.add_argument(
        '--device',
        type=str,
        default='mps:0',
    )

    parser.add_argument(
        '--training-dataset-path',
        type=str,
        default='data/TinyStoriesV2-GPT4-train-tokenized.npy',
    )

    parser.add_argument(
        '--validation-dataset-path',
        type=str,
        default='data/TinyStoriesV2-GPT4-valid-tokenized.npy',
    )

    parser.add_argument(
        '--vocab-size',
        type=int,
        default=10000,
    )

    parser.add_argument(
        '--context-length',
        type=int,
        default=256,
    )

    parser.add_argument(
        '--d-model',
        type=int,
        default=512,
    )

    parser.add_argument(
        '--d-ff',
        type=int,
        default=1344,
    )

    parser.add_argument(
        '--rope-theta',
        type=int,
        default=10000,
    )

    parser.add_argument(
        '--num-layers',
        type=int,
        default=4,
    )

    parser.add_argument(
        '--num-heads',
        type=int,
        default=16,
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
    )

    parser.add_argument(
        '--num-tokens-processed',
        type=int,
        # default=327680000,
        default=40000000,
    )

    parser.add_argument(
        '--learning-rate',
        type=str,
        default='1e-3',
    )

    parser.add_argument(
        '--optimizer-weight-decay',
        type=str,
        default='0.01',
    )

    parser.add_argument(
        '--optimizer-beta1',
        type=str,
        default='0.9',
    )

    parser.add_argument(
        '--optimizer-beta2',
        type=str,
        default='0.9999',
    )

    parser.add_argument(
        '--optimizer-eps',
        type=str,
        default='1e-8',
    )

    return  parser.parse_args()


def read_dataset_from_np(path: str):
    data = np.load(path, mmap_mode='r')
    return data


def model_train(args):
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

    optimizer = AdamW(
        params=model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.optimizer_weight_decay),
        betas=(float(args.optimizer_beta1), float(args.optimizer_beta2)),
        eps=float(args.optimizer_eps),
    )

    training_data = read_dataset_from_np(args.training_dataset_path)
    validation_data = read_dataset_from_np(args.validation_dataset_path)

    total_steps = args.num_tokens_processed // args.batch_size // args.context_length
    print(f'total_steps = {total_steps}')

    for step_cnt in range(total_steps):
        features, label = get_batch(
            dataset=training_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device)

        train_pred = model(features)
        train_loss = cross_entropy_loss(train_pred.view(-1, train_pred.size(-1)), label.view(-1))
        print(f'Step {step_cnt}: train loss = {train_loss}')
        
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step_cnt % 100 == 0:
            run_validation(model)


if __name__ == "__main__":
    args = parse_args()

    model_train(args)