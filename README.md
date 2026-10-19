# Hydro-DP-Model

PyTorch implementation of five conceptual hydrological models (XAJ, HBV, SAC-SMA, GR4J and Tank) with Single, Cross and Joint training strategies.

## Data

Meteorological inputs, catchment descriptors and model attributes are included. Restricted gauge discharge is excluded. Add authorized daily discharge (m3/s) for 2000-2019 to:

- `data/js/observation/Q.csv`
- `data/hhy/observation/Q.csv`

Both files contain only their original headers. Preserve column names and leave missing observations blank.

## Run

Run from the repository root after adding discharge data:

```bash
python -m pip install -r requirements.txt
python train_3schemes_5models.py
python simulate_3schemes_5models.py
```

Training: 2000-2012; validation: 2013-2015; testing: 2016-2019; warm-up: 1999. Cross uses source-basin training and target-basin validation for model selection.

Source code and configurations are unchanged. Upload only the header-only discharge files to GitHub.
