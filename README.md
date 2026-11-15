# MedVision AI

Full-stack X-ray analysis application with chest and musculoskeletal models, Grad-CAM visualizations, and downloadable reports. The project is intended for education and research, not clinical diagnosis or medical decision-making.

## Features

- Chest image analysis, normal/abnormal assessment, and multi-label classification.
- Musculoskeletal body-part classification and abnormality detection.
- Confidence scores and Grad-CAM visualizations.
- Structured interpretation and downloadable PDF reports.

## Models and datasets

| Component | Dataset or architecture | Recorded validation metric |
| --- | --- | --- |
| Chest model | CheXpert, DenseNet-121 | See the chest model files |
| Bone body-part classifier | MURA | Accuracy: 96.7% |
| Bone abnormality classifier | MURA | AUC: 0.871 |

Dataset references: CheXpert from the Stanford ML Group and MURA from Stanford. These figures describe model validation, not clinical performance.

## Structure

| Directory | Purpose |
| --- | --- |
| `frontend/` | Next.js, React, TypeScript, and Tailwind interface |
| `backend/` | FastAPI service, model inference, and reporting |
| `ai/` | Chest and bone modeling code and validation metrics |
| `models/` | Trained model weights |

The model stack uses PyTorch, Torchvision, and Grad-CAM. Hosting configuration targets Vercel for the frontend and Railway for the backend.

## Local development

Frontend:

```sh
cd frontend
npm install
npm run dev
```

Backend, in a separate terminal:

```sh
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

## Analysis flow

1. Upload a chest or bone X-ray.
2. Validate the image and select the appropriate model.
3. Run inference and generate scores.
4. Generate a Grad-CAM visualization.
5. Display results and provide a report download.

## Planned work

Additional imaging modalities, inference performance, authentication, patient history, and longitudinal comparison.

## Links

- [Web application](https://med-vision-maoss267a-leya3.vercel.app/)
- [Repository](https://github.com/lrglazer/medVision_AI)

## Author

Leya Glazer, University of Texas at Dallas.

## License and acknowledgments

This project is licensed under the MIT License. See the repository's license notice.

Acknowledgments: Stanford ML Group, Stanford AIMI, PyTorch, FastAPI, Next.js, Vercel, and Railway.
