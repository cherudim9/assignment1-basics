uv run python3 cs336_basics/model_train.py \
-p 'my_owt' \
-r 'lr1e-3 bs128' \
--root-folder /workspace/model \
--model-folder lr1e3bs128 \
--device cuda:0 \
--training-dataset-path /workspace/data/owt_train-tokenized-local.npy \
--validation-dataset-path /workspace/data/owt_valid-tokenized-local.npy \
--vocab-size 32000 \
--context-length 128 \
--d-model 512 \
--d-ff 1344 \
--batch-size 256 \
--num-tokens-processed 327680000 \
--max-learning-rate '1e-3' \
--min-learning-rate '1e-4' \


uv run python3 cs336_basics/model_train.py \
-p 'my_owt' \
-r 'lr1e-3 bs128' \
--root-folder /workspace/model \
--model-folder lr1e3bs256 \
--device cuda:0 \
--training-dataset-path /workspace/data/owt_train-tokenized-local.npy \
--validation-dataset-path /workspace/data/owt_valid-tokenized-local.npy \
--vocab-size 32000 \
--context-length 128 \
--d-model 512 \
--d-ff 1344 \
--batch-size 256 \
--num-tokens-processed 16384000 \
--max-learning-rate '1e-3' \
--min-learning-rate '1e-4' \
--warmup-tiers 500


# LR

uv run python3 cs336_basics/model_train.py \
-p 'my_owt' \
-r 'lr1e-3 bs256' \
--root-folder /workspace/model \
--model-folder lr1e3bs256 \
--device cuda:0 \
--training-dataset-path /workspace/data/owt_train-tokenized-local.npy \
--validation-dataset-path /workspace/data/owt_valid-tokenized-local.npy \
--vocab-size 32000 \
--context-length 128 \
--d-model 512 \
--d-ff 1344 \
--batch-size 256 \
--num-tokens-processed 16384000 \
--max-learning-rate '1e-3' \
--min-learning-rate '1e-4'

uv run python3 cs336_basics/model_train.py \
-p 'my_owt' \
-r 'lr2.5e-3 bs256' \
--root-folder /workspace/model \
--model-folder lr1e25bs256 \
--device cuda:0 \
--training-dataset-path /workspace/data/owt_train-tokenized-local.npy \
--validation-dataset-path /workspace/data/owt_valid-tokenized-local.npy \
--vocab-size 32000 \
--context-length 128 \
--d-model 512 \
--d-ff 1344 \
--batch-size 256 \
--num-tokens-processed 16384000 \
--max-learning-rate '2.5e-3' \
--min-learning-rate '2.5e-4'

uv run python3 cs336_basics/model_train.py \
-p 'my_owt' \
-r 'lr5e-3 bs256' \
--root-folder /workspace/model \
--model-folder lr5e3bs256 \
--device cuda:0 \
--training-dataset-path /workspace/data/owt_train-tokenized-local.npy \
--validation-dataset-path /workspace/data/owt_valid-tokenized-local.npy \
--vocab-size 32000 \
--context-length 128 \
--d-model 512 \
--d-ff 1344 \
--batch-size 256 \
--num-tokens-processed 16384000 \
--max-learning-rate '5e-3' \
--min-learning-rate '5e-4'

uv run python3 cs336_basics/model_train.py \
-p 'my_owt' \
-r 'lr5e-4 bs256' \
--root-folder /workspace/model \
--model-folder lr5e4bs256 \
--device cuda:0 \
--training-dataset-path /workspace/data/owt_train-tokenized-local.npy \
--validation-dataset-path /workspace/data/owt_valid-tokenized-local.npy \
--vocab-size 32000 \
--context-length 128 \
--d-model 512 \
--d-ff 1344 \
--batch-size 256 \
--num-tokens-processed 16384000 \
--max-learning-rate '5e-4' \
--min-learning-rate '5e-5'

uv run python3 cs336_basics/model_train.py \
-p 'my_owt' \
-r 'lr7.5e-4 bs256' \
--root-folder /workspace/model \
--model-folder lr75e4bs256 \
--device cuda:0 \
--training-dataset-path /workspace/data/owt_train-tokenized-local.npy \
--validation-dataset-path /workspace/data/owt_valid-tokenized-local.npy \
--vocab-size 32000 \
--context-length 128 \
--d-model 512 \
--d-ff 1344 \
--batch-size 256 \
--num-tokens-processed 16384000 \
--max-learning-rate '7.5e-4' \
--min-learning-rate '7.5e-5'

## conclusion 
--max-learning-rate '2.5e-3' \
--min-learning-rate '2.5e-4'

# BS

uv run python3 cs336_basics/model_train.py \
-p 'my_owt' \
-r 'lr2.5e-3 bs64' \
--root-folder /workspace/model \
--model-folder lr25e3bs64 \
--device cuda:0 \
--training-dataset-path /workspace/data/owt_train-tokenized-local.npy \
--validation-dataset-path /workspace/data/owt_valid-tokenized-local.npy \
--vocab-size 32000 \
--context-length 128 \
--d-model 512 \
--d-ff 1344 \
--batch-size 64 \
--num-tokens-processed 16384000 \
--max-learning-rate '2.5e-3' \
--min-learning-rate '2.5e-4'

uv run python3 cs336_basics/model_train.py \
-p 'my_owt' \
-r 'lr2.5e-3 bs128' \
--root-folder /workspace/model \
--model-folder lr25e3bs128 \
--device cuda:0 \
--training-dataset-path /workspace/data/owt_train-tokenized-local.npy \
--validation-dataset-path /workspace/data/owt_valid-tokenized-local.npy \
--vocab-size 32000 \
--context-length 128 \
--d-model 512 \
--d-ff 1344 \
--batch-size 128 \
--num-tokens-processed 16384000 \
--max-learning-rate '2.5e-3' \
--min-learning-rate '2.5e-4'

uv run python3 cs336_basics/model_train.py \
-p 'my_owt' \
-r 'lr2.5e-3 bs192' \
--root-folder /workspace/model \
--model-folder lr25e3bs192 \
--device cuda:0 \
--training-dataset-path /workspace/data/owt_train-tokenized-local.npy \
--validation-dataset-path /workspace/data/owt_valid-tokenized-local.npy \
--vocab-size 32000 \
--context-length 128 \
--d-model 512 \
--d-ff 1344 \
--batch-size 192 \
--num-tokens-processed 16384000 \
--max-learning-rate '2.5e-3' \
--min-learning-rate '2.5e-4'

uv run python3 cs336_basics/model_train.py \
-p 'my_owt' \
-r 'lr2.5e-3 bs512' \
--root-folder /workspace/model \
--model-folder lr25e3bs512 \
--device cuda:0 \
--training-dataset-path /workspace/data/owt_train-tokenized-local.npy \
--validation-dataset-path /workspace/data/owt_valid-tokenized-local.npy \
--vocab-size 32000 \
--context-length 128 \
--d-model 512 \
--d-ff 1344 \
--batch-size 512 \
--num-tokens-processed 16384000 \
--max-learning-rate '2.5e-3' \
--min-learning-rate '2.5e-4'

uv run python3 cs336_basics/model_train.py \
-p 'my_owt' \
-r 'lr2.5e-3 bs384' \
--root-folder /workspace/model \
--model-folder lr25e3bs384 \
--device cuda:0 \
--training-dataset-path /workspace/data/owt_train-tokenized-local.npy \
--validation-dataset-path /workspace/data/owt_valid-tokenized-local.npy \
--vocab-size 32000 \
--context-length 128 \
--d-model 512 \
--d-ff 1344 \
--batch-size 384 \
--num-tokens-processed 16384000 \
--max-learning-rate '2.5e-3' \
--min-learning-rate '2.5e-4'


# context length

uv run python3 cs336_basics/model_train.py \
-p 'my_owt' \
-r 'lr2.5e-3 bs128 cl192' \
--root-folder /workspace/model \
--model-folder lr25e3bs128cl192 \
--device cuda:0 \
--training-dataset-path /workspace/data/owt_train-tokenized-local.npy \
--validation-dataset-path /workspace/data/owt_valid-tokenized-local.npy \
--vocab-size 32000 \
--context-length 192 \
--d-model 512 \
--d-ff 1344 \
--batch-size 128 \
--num-tokens-processed 16384000 \
--max-learning-rate '2.5e-3' \
--min-learning-rate '2.5e-4'

uv run python3 cs336_basics/model_train.py \
-p 'my_owt' \
-r 'lr2.5e-3 bs128 cl160' \
--root-folder /workspace/model \
--model-folder lr25e3bs128cl160 \
--device cuda:0 \
--training-dataset-path /workspace/data/owt_train-tokenized-local.npy \
--validation-dataset-path /workspace/data/owt_valid-tokenized-local.npy \
--vocab-size 32000 \
--context-length 160 \
--d-model 512 \
--d-ff 1344 \
--batch-size 128 \
--num-tokens-processed 16384000 \
--max-learning-rate '2.5e-3' \
--min-learning-rate '2.5e-4'

uv run python3 cs336_basics/model_train.py \
-p 'my_owt' \
-r 'lr2.5e-3 bs128 cl256' \
--root-folder /workspace/model \
--model-folder lr25e3bs128cl256 \
--device cuda:0 \
--training-dataset-path /workspace/data/owt_train-tokenized-local.npy \
--validation-dataset-path /workspace/data/owt_valid-tokenized-local.npy \
--vocab-size 32000 \
--context-length 256 \
--d-model 512 \
--d-ff 1344 \
--batch-size 128 \
--num-tokens-processed 16384000 \
--max-learning-rate '2.5e-3' \
--min-learning-rate '2.5e-4'



uv run python3 cs336_basics/model_train.py \
-p 'my_owt' \
-r 'full - lr2.5e-3 bs128 cl192' \
--root-folder /workspace/model \
--model-folder lr25e3bs128cl192 \
--device cuda:0 \
--training-dataset-path /workspace/data/owt_train-tokenized-local.npy \
--validation-dataset-path /workspace/data/owt_valid-tokenized-local.npy \
--vocab-size 32000 \
--context-length 192 \
--d-model 512 \
--d-ff 1344 \
--batch-size 128 \
--num-tokens-processed 80000000 \
--max-learning-rate '2.5e-3' \
--min-learning-rate '2.5e-4'


uv run python3 cs336_basics/model_train.py \
-p 'my_owt' \
-r 'full2 - lr2.5e-3 bs192 cl60' \
--root-folder /workspace/model \
--model-folder lr25e3bs192cl160 \
--device cuda:0 \
--training-dataset-path /workspace/data/owt_train-tokenized-local.npy \
--validation-dataset-path /workspace/data/owt_valid-tokenized-local.npy \
--vocab-size 32000 \
--context-length 160 \
--d-model 512 \
--d-ff 1344 \
--batch-size 192 \
--num-tokens-processed 80000000 \
--max-learning-rate '2.5e-3' \
--min-learning-rate '2.5e-4'

uv run python3 cs336_basics/model_train.py \
-p 'my_owt' \
-r 'full - lr2.5e-3 bs256 c128' \
--root-folder /workspace/model \
--model-folder lr25e3bs256cl128full \
--device cuda:0 \
--training-dataset-path /workspace/data/owt_train-tokenized-local.npy \
--validation-dataset-path /workspace/data/owt_valid-tokenized-local.npy \
--vocab-size 32000 \
--context-length 128 \
--d-model 512 \
--d-ff 1344 \
--batch-size 256 \
--num-tokens-processed 100000000 \
--max-learning-rate '2.5e-3' \
--min-learning-rate '2.5e-4'

uv run python3 cs336_basics/model_train.py \
-p 'my_owt' \
-r 'full - lr2.5e-3 to 1.25e-3 bs256 c128' \
--root-folder /workspace/model \
--model-folder lr25e3to1.25e3bs256cl128full \
--device cuda:0 \
--training-dataset-path /workspace/data/owt_train-tokenized-local.npy \
--validation-dataset-path /workspace/data/owt_valid-tokenized-local.npy \
--vocab-size 32000 \
--context-length 128 \
--d-model 512 \
--d-ff 1344 \
--batch-size 256 \
--num-tokens-processed 110000000 \
--max-learning-rate '2.5e-3' \
--min-learning-rate '1.25e-3'


uv run python3 cs336_basics/model_train.py \
-p 'my_owt' \
-r 'full - lr7.5e-3 bs256 c128' \
--root-folder /workspace/model \
--model-folder lr75e3bs256cl128full \
--device cuda:0 \
--training-dataset-path /workspace/data/owt_train-tokenized-local.npy \
--validation-dataset-path /workspace/data/owt_valid-tokenized-local.npy \
--vocab-size 32000 \
--context-length 128 \
--d-model 512 \
--d-ff 1344 \
--batch-size 256 \
--num-tokens-processed 110000000 \
--max-learning-rate '7.5e-3' \
--min-learning-rate '7.5e-4'
