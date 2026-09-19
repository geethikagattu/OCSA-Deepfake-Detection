# OCSA Deepfake Detection

A controlled baseline for building and evaluating an online adaptation and OCSA decision pipeline for deepfake detection.

## Current pipeline

```text
Dataset -> Face preprocessing -> Baseline Deepfake Detector -> Evaluate -> Save results
```

The first baseline uses a pretrained PyTorch ResNet18 with a binary classifier head. The initial notebook only verifies the environment and loads the model; it does not train or download a dataset.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run `notebooks/01_baseline.ipynb` locally or in Google Colab. In Colab, update `YOUR_GITHUB_REPOSITORY_URL` in the first cell before running it.

## Repository layout

- `src/detector`: baseline ResNet18 detector
- `src/adaptation`: adaptation strategy interface and WAIT strategy
- `src/ocsa`: OCSA decision-layer and adaptation history skeleton
- `src/evaluation`: binary classification metrics
- `configs`: experiment configuration
- `notebooks`: reproducible experiment entry points
- `data`: dataset documentation and local data location
- `experiments/results`: generated result files

## Current stopping point

Do not download or train on a dataset yet. First confirm that all team members can install the requirements, run the baseline notebook, and see the selected device and successful model load. Dataset selection, splits, preprocessing, and evaluation protocol should then be agreed on before experimentation.
