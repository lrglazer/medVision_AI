cd C:\Users\Owner\Documents\Medvision

& ".\ai\chest\.venv\Scripts\python.exe" -m ai.shared.train_validator `
  --name bone_xray `
  --positive-dir ".\datasets\MURA-v1.1\train" `
  --negative-dir ".\datasets\CheXpert-v1.0-small\train" `
  --negative-dir ".\datasets\validator_negatives\general" `
  --output ".\ai\bone\models\best_bone_validator.pt" `
  --epochs 3 `
  --batch-size 64
