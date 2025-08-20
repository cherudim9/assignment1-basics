import argparse
from argparse import ArgumentParser
import numpy as np
import numpy.typing as npt
import os
from pathlib import Path
import time
import torch

from cs336_basics.args import get_parser
from cs336_basics.model.data import get_batch
from cs336_basics.model.model import TransformerLm
from cs336_basics.model.nn_utils import cross_entropy_loss
from cs336_basics.model.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.model.serialization import load_checkpoint, save_checkpoint


VALIDATION_PERCENTAGE = 0.1


def read_dataset_from_np(path: str) -> npt.NDArray:
    data = np.load(path, mmap_mode='r')
    return data


def display_model_info(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'total_params = {total_params}')
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'trainable_params = {trainable_params}')
    for name, module in model.named_modules():
        layer_params = sum(p.numel() for p in module.parameters())
        print(f"{name}, Parameters: {layer_params}")


def adjust_learning_rate(
    args: ArgumentParser,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    step_cnt: int
):
    current_lr = get_lr_cosine_schedule(
        it=step_cnt,
        max_learning_rate=float(args.max_learning_rate),
        min_learning_rate=float(args.min_learning_rate),
        warmup_iters=args.warmup_tiers,
        cosine_cycle_iters=total_steps-args.warmup_tiers,
    )

    for group in optimizer.param_groups:
        group['lr'] = current_lr


def run_validation(
    args: ArgumentParser,
    current_step: int,
    model: torch.nn.Module,
    dataset: npt.NDArray,
):
    start_time = time.time()
    total_steps = int(dataset.shape[0] * VALIDATION_PERCENTAGE) // (args.context_length * args.batch_size)
    losses = []
    for _ in range(total_steps):
        with torch.no_grad():
            features, label = get_batch(
                dataset=dataset,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device)
        valid_pred = model(features)
        valid_loss = cross_entropy_loss(valid_pred.view(-1, valid_pred.size(-1)), label.view(-1))
        losses.append(valid_loss.item())

    loss = np.mean(losses)
    duration = time.time() - start_time
    processed_tokens = total_steps * args.batch_size * args.context_length
    print(f'validation loss at step {current_step}: {loss: .6f} (running time = {duration: .2f} seconds with {processed_tokens} tokens).')


def model_train(args: ArgumentParser):
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

    if args.device == 'mps':
        model = torch.compile(model, backend="aot_eager")

    display_model_info(model)

    optimizer = AdamW(
        params=model.parameters(),
        lr=float(args.max_learning_rate),
        weight_decay=float(args.optimizer_weight_decay),
        betas=(float(args.optimizer_beta1), float(args.optimizer_beta2)),
        eps=float(args.optimizer_eps),
    )

    training_data = read_dataset_from_np(args.training_dataset_path)
    validation_data = read_dataset_from_np(args.validation_dataset_path)
    print(f'length of training_data = {training_data.shape}')
    print(f'length of validation_data = {validation_data.shape}')

    total_steps = args.num_tokens_processed // args.batch_size // args.context_length
    print(f'training total steps = {total_steps}')

    for step_cnt in range(total_steps):
        features, label = get_batch(
            dataset=training_data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=args.device)

        train_pred = model(features)
        train_loss = cross_entropy_loss(train_pred.view(-1, train_pred.size(-1)), label.view(-1))
        
        adjust_learning_rate(args, optimizer, total_steps, step_cnt)
        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step_cnt % 50 == 0:
            print(f'Step {step_cnt}: train loss = {train_loss: .6f}, learning rate = {optimizer.param_groups[0]['lr']: .10f}')

        if step_cnt % 100 == 0:
            run_validation(args, step_cnt, model, validation_data)
    
    model_path = Path(args.root_folder) / Path(args.model_folder)
    os.makedirs(model_path, exist_ok=True)
    model_dest_path = model_path / f"model-checkpoint-step-{total_steps}"
    print(f'saving model to {model_dest_path} ...')
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        iteration=total_steps,
        out=model_dest_path)


if __name__ == "__main__":
    args = get_parser("basic model trainier").parse_args()

    model_train(args)