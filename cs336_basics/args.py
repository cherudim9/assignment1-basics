import argparse


def get_parser(description: str):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '-p',
        '--wandb-project-name',
        type=str,
        default='tiny-stories-first',
    )

    parser.add_argument(
        '-r',
        '--wandb-run-name',
        type=str,
        default='',
    )

    parser.add_argument(
        '--wandb-group-name',
        type=str,
        default='',
    )

    parser.add_argument(
        '--root-folder',
        type=str,
        default='/Users/haomin/Documents/CS336/assignment1-basics',
    )

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
        '--max-learning-rate',
        type=str,
        default='1e-3',
    )

    parser.add_argument(
        '--min-learning-rate',
        type=str,
        default='1e-4',
    )

    parser.add_argument(
        '--warmup-tiers',
        type=int,
        default=100,
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
        default='0.95',
    )

    parser.add_argument(
        '--optimizer-eps',
        type=str,
        default='1e-8',
    )

    return  parser
