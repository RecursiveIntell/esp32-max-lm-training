# ESP32-S3 character-LSTM training

A PyTorch training script for a character-level LSTM intended for ESP32-S3 inference experiments. It combines TinyStories text with generated sensor/status phrases, trains a configurable embedding/LSTM/output-head model, and exports checkpoints, samples, and Rust weight constants.

The script selects CUDA when available and otherwise uses CPU. This repository contains the training/export script; it does not contain ESP32 firmware or prove that every chosen model fits or runs on a board.

## Setup

Use a Python environment with NumPy and a PyTorch build appropriate for your host:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy torch
python train_esp32_max_lstm.py --help
```

## Training

The defaults use hidden size 512, three layers, and 6,000 training steps. Inspect resource use and choose a new output directory before starting:

```bash
python train_esp32_max_lstm.py \
  --hidden 512 --layers 3 --steps 6000 \
  --out runs/esp32s3-h512-experiment
```

On first use the script downloads `TinyStories-train.txt` from the upstream Hugging Face dataset if its local cache is absent. `--max-corpus-chars` limits the text read for training, not the size of the initial dataset download. Review dataset terms and allow sufficient disk space before running.

Additional options include sequence length, batch size, learning-rate schedule, corpus size, and generated edge-domain text size. The [argument definitions](train_esp32_max_lstm.py) are the current interface.

## Outputs and interpretation

The selected output directory receives checkpoints such as `final.pt`, training summaries, samples, and `weights/esp32s3_max_lstm_weights.rs`. The exporter quantizes weight arrays to int8 with scales and preserves floating-point bias data in Rust source. It does not emit a flash-ready RILM binary by itself.

This model emits characters, not BPE tokens. Training loss, generated sample text, and parameter count do not establish on-device throughput, memory fit, firmware compatibility, or correctness outside the trained prompt distribution. Validate exported weights against the exact inference runtime before deployment.

## Source checks

```bash
python -m py_compile train_esp32_max_lstm.py
```

This syntax check does not execute training, download the dataset, or verify a hardware deployment.
