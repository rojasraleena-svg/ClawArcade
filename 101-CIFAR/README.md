# Problem brief

Train a fixed **SmallCNN** on **CIFAR-10**. You only control hyperparameters and `--epochs` via the CLI; **one run, one score**. The script evaluates on the **official test split** (`train=False`) at an increasingly dense set of epochs (sparse early, every epoch in the last five). **`--epochs` is capped at 80** (minimum 1).

RNG seed, DataLoader workers, data directory, and device are **fixed globals** in `train.py` (`SEED`, `NUM_WORKERS`, `DATA_ROOT`, `DEVICE`). `train.py` does **not** write any log or metrics files (stdout only).

## Environment

- Python **3.10+**
- From this directory:

```bash
cd 101-CIFAR
uv sync
# or: pip install -e .
```

CIFAR-10 is downloaded automatically on first run under `DATA_ROOT` in `train.py` (default `./data`).

## How to run

```bash
python train.py [OPTIONS...]
# e.g. with uv: uv run python train.py [OPTIONS...]
```

| Hyperparameter | Flag | Default | Notes |
|----------------|------|---------|--------|
| Learning rate | `--lr` | `0.001` | |
| Weight decay | `--weight-decay` | `0.0` | |
| Batch size | `--batch-size` | `128` | |
| Epochs | `--epochs` | `10` | **1–80** |
| SGD momentum | `--momentum` | `0.9` | |

Example:

```bash
python train.py --epochs 40 --lr 0.01 --batch-size 128
```

## Expected output format

### Stdout (always exactly two lines)

**Line 1:** comma-separated **evaluation epoch indices** (test set evaluated after each listed epoch, in order).

**Line 2:** comma-separated **test accuracies** at those epochs (same count as line 1; four decimal places).

Example for **`epochs = 40`**:

```text
1,10,20,25,30,35,36,37,38,39,40
0.3521,0.5120,0.5834,0.6012,0.6123,0.6189,0.6201,0.6210,0.6215,0.6220,0.6224
```

If a run ends early (e.g. OOM), line 1 / line 2 reflect only checkpoints **completed** before failure; either line may be empty. Process exit code is **0** on success, **1** on failure.
