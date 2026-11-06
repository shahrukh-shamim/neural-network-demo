# Neural network movie demo

A small Python learning project that trains a single sigmoid neuron using movie
length (hours) and IMDb rating. It currently trains on one positive example;
its output is an educational demonstration, not a reliable recommendation.

## Layout

```text
scripts/
  train.py                 Training command
  predict.py               Prediction command
data/
  weights.json             Working weights updated by training
  original-weights.json    Preserved starting-weight snapshot
  final-weights.json       Separate weights used by prediction
  favourite-movie.json     Movie input used by prediction
```

## Run

Python 3 is required. The current scripts use only the Python standard library.
From the project root:

```sh
python scripts/predict.py
python scripts/train.py
```

Data paths are resolved relative to the scripts, so the commands also work when
invoked by absolute path from another directory.

Training updates `data/weights.json` after each step. Prediction reads
`data/final-weights.json`; it does not automatically use the latest training
weights. The original and final snapshots remain separate intentionally.

## Upcoming web integration

Before connecting Flask, extract the model calculations and validation into an
importable Python package and keep these scripts as command-line entry points.
The scripts currently execute when loaded, so they should not be imported into
a web server yet. Flask routes, templates, and static assets can then be added
alongside the shared model code.
