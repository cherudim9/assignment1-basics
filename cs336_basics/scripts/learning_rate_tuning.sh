uv run python3 cs336_basics/model_train.py --max-learning-rate '1e-2' --min-learning-rate '1e-3' -r 'lr1e-2' --model-folder 'data/model_tiny_stories/lr1e2' &&

uv run python3 cs336_basics/model_train.py --max-learning-rate '5e-3' --min-learning-rate '5e-4' -r 'lr5e-3' --model-folder 'data/model_tiny_stories/lr5e3' &&

uv run python3 cs336_basics/model_train.py --max-learning-rate '1e-4' --min-learning-rate '1e-5' -r 'lr1e-4' --model-folder 'data/model_tiny_stories/lr1e4' &&

uv run python3 cs336_basics/model_train.py --max-learning-rate '2e-3' --min-learning-rate '2e-4' -r 'lr2e-3' --model-folder 'data/model_tiny_stories/lr2e3'