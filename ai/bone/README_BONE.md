# MedVision Bone Specialist

## Folder location

Place these files in:

```text
C:\Users\Owner\Documents\Medvision\ai\bone
```

Expected dataset location:

```text
C:\Users\Owner\Documents\Medvision\datasets\MURA-v1.1
```

Expected dataset contents:

```text
MURA-v1.1
├── train
├── valid
├── train_image_paths.csv
├── train_labeled_studies.csv
├── valid_image_paths.csv
└── valid_labeled_studies.csv
```

## Commands

From:

```text
C:\Users\Owner\Documents\Medvision\ai\bone
```

activate the existing chest environment:

```powershell
& "..\chest\.venv\Scripts\Activate.ps1"
```

Validate the dataset:

```powershell
python prepare_mura.py
```

Train:

```powershell
python train_bone.py
```

Resume after interruption:

```powershell
python train_bone.py --resume
```

Evaluate and select the decision threshold:

```powershell
python evaluate_bone.py
```

Outputs:

```text
models\best_mura_abnormality.pt
models\mura_validation_metrics.json
models\mura_threshold.json
checkpoints\last.pt
checkpoints\epoch_1.pt
...
```
