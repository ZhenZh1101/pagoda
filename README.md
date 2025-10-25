# Pagoda

Train small English language models from scratch on Apple Silicon with MLX. Includes data preparation, tokenizer training, pretraining, supervised fine-tuning (SFT), evaluation, and chat.

## Setup

Requires an Apple Silicon Mac, Python 3.11 or 3.12, `uv`, and `tmux`. Reserve around 100 GB of free disk space for data and checkpoints.

Clone or download this repository, then run:

```bash
cd pagoda
uv sync
```

## Prepare data

Run these steps before training:

```bash
uv run python -m scripts.download
uv run python -m scripts.prepare_data
uv run python -m scripts.tok_train --vocab 32768
uv run python -m scripts.build_shards
uv run python -m scripts.verify_data
```

Datasets, trained tokenizers, model weights, and logs are generated locally and are not included in this repository.

## Train and chat

```bash
bash runs/train_24h.sh
```

This trains a 55M-parameter model with a target budget of 24 hours for pretraining, SFT, and evaluation. Follow progress in `logs/d512_24h/pipeline/pipeline.log`.

After training:

```bash
uv run python -m scripts.chat --run d512_24h --source sft --interactive
```

Larger models use `runs/train_3d.sh` or `runs/train_5d.sh`, followed by `RUN=<run-name> bash runs/after_pretrain.sh` for SFT and evaluation.

## Development

Run tests with `uv run pytest -q`. Core code lives in `pagoda/`, commands in `scripts/`, and data mixtures in `configs/`.

See the [design plan](pagoda_v0.1_plan.md) for architecture and training details.
