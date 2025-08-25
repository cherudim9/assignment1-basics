uv run python3 cs336_basics/model_train.py \
--max-learning-rate '1e-3' \
--min-learning-rate '1e-4' \
--wandb-group-name 'batch-size-tuning' \
--batch-size 4 \
-r 'batch-size-4' \
--model-folder \
'data/model_tiny_stories/bs4' &&

uv run python3 cs336_basics/model_train.py \
--max-learning-rate '1e-3' \
--min-learning-rate '1e-4' \
--wandb-group-name 'batch-size-tuning' \
--batch-size 16 \
-r 'batch-size-16' \
--model-folder \
'data/model_tiny_stories/bs16' &&

uv run python3 cs336_basics/model_train.py \
--max-learning-rate '1e-3' \
--min-learning-rate '1e-4' \
--wandb-group-name 'batch-size-tuning' \
--batch-size 64 \
-r 'batch-size-64' \
--model-folder \
'data/model_tiny_stories/bs64'


uv run python3 cs336_basics/model_train.py \
--max-learning-rate '1e-1' \
--min-learning-rate '1e-2' \
--batch-size 64 \
-r 'gradient-clipping' \
--model-folder \
'data/model_tiny_stories/gradient'
