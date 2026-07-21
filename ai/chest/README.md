# MedVision CheXpert Multi-Label Model

This package converts the original binary pneumonia pipeline into a
14-observation CheXpert training pipeline while leaving the working model intact.

## Expected data layout

data/
└── CheXpert-v1.0-small/
    ├── train.csv
    ├── valid.csv
    ├── train/
    └── valid/

## Run

python -m pip install -r requirements_multilabel.txt
python prepare_chexpert.py
python train_multilabel.py
python evaluate_multilabel.py

## Outputs

models/
├── best_chexpert_multilabel.pt
├── chexpert_thresholds.json
└── chexpert_validation_metrics.csv

Each finding receives its own validation-derived threshold and a three-state
report result: negative model finding, indeterminate, or positive model finding.
These are research outputs, not confirmed diagnoses.
