# FAST-STG

FAST-STG is a framework for probabilistic spatiotemporal forecasting. It learns conditional location and scale, then transports standardized residuals with an SDE sampler.

## Environment

The source was tested with Python 3.9.25 and PyTorch 2.3.1+cu121. Package versions are recorded in `requirements.txt`. Use a separate environment; install a PyTorch build suitable for your device if necessary.

```bash
conda create -n fast-stg python=3.9
conda activate fast-stg
python -m pip install -r requirements.txt
```

NNI and TensorBoard are optional integrations and are not required for the default commands. The recorded environment has passed unit tests; installation in a brand-new environment has not yet been verified.

## Data

Supported datasets are `AIR_BJ`, `AIR_GZ`, `PEMS03`, `PEMS04`, `PEMS07`, and `PEMS08`. Obtain and prepare the data separately. See [data/README.md](data/README.md) for the required files, shapes, and ordering.

## Training

Run commands from the repository root after preparing the selected dataset. Explicit device selection avoids reliance on the legacy GPU auto-dispatch helper.

```bash
python train.py --data PEMS08 --device cuda:0
python tools/run.py --task main --data AIR_BJ --device cuda:0 --execute
```

Use `--device cpu` for CPU execution. Training writes checkpoints, logs, and forecasts under `output/`. The unified runner also writes JSON metrics beneath `output/metrics/`.

The default configuration uses 12 history steps, 12 prediction steps, batch size 8, learning rate 0.002, 5 location/scale pretraining epochs, 20 sampling steps, and 100 samples for the main experiment. The noise scale is 0.08 for AIR datasets and 0.05 for PEMS datasets. The maximum training budget is 300 epochs with patience 10. The experiment settings in `train.py` define the supported parameter overrides.

## Code and license

`train.py` provides training, evaluation, default experiment settings, and checkpoint compatibility checks. `algorithm/rf_sde/model.py` implements FAST-STG. The retained `algorithm/diffstg/` files provide the shared UGnet backbone and dependencies used by the entry point. `utils/` provides metrics and utilities.

This release is based on DiffSTG. The upstream copyright notice and MIT license are preserved in [LICENSE](LICENSE). This repository contains source code and tests. Obtain data separately; training generates model checkpoints and experiment outputs.
