# Data input format

Prepare each supported dataset separately under this repository:

```text
data/dataset/<DATASET>/
├── flow.npy
└── adj.npy
```

`<DATASET>` is one of `AIR_BJ`, `AIR_GZ`, `PEMS03`, `PEMS04`, `PEMS07`, or `PEMS08`.

- `flow.npy`: a numeric NumPy array with shape `(T, V, C)`, where timestamps are in chronological order, `V` is the number of nodes, and `C >= 1`. The current AIR/PEMS loader uses channel 0. Store unnormalized measurements; preprocessing fits normalization on the training portion. Use the manuscript target variable as channel 0.
- `adj.npy`: a numeric NumPy array with shape `(V, V)`. Its row/column order must match the node order in `flow.npy`; preserve the graph used for the manuscript experiments.
- AIR missing targets may be represented by `NaN`. The default input filling is causal forward fill with leading missing values filled by zero; unobserved targets are excluded from losses and metrics.
- Time boundaries are `int(0.6 * T)` and `int(0.8 * T)`. Do not shuffle timestamps or normalize across the whole dataset before saving these arrays.

For a PEMS archive whose `data` field is already ordered as `(T, V, C)` with the desired target in channel 0, this conversion illustrates the input format:

```python
from pathlib import Path
import numpy as np

source = Path("PATH_TO_DATASET.npz")
destination = Path("data/dataset/PEMS04")
destination.mkdir(parents=True, exist_ok=True)
with np.load(source, allow_pickle=False) as archive:
    flow = archive["data"][..., 0:1].astype(np.float32)
np.save(destination / "flow.npy", flow)
# Prepare adj.npy separately using the experiment's graph and matching node IDs.
```

The archive field name, feature meaning, node mapping, date range, sampling rate, and graph construction must be checked against the dataset actually used in the manuscript. The example does not reconstruct the graph or prepare AIR data.

This source-only package does not include data, download links, or a complete raw-data preparation pipeline. Obtain the same data and graph used for the experiments; dataset provenance and exact acquisition/conversion instructions still need to be documented before claiming end-to-end reproducibility.
