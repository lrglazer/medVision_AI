MedVision AI

MedVision AI is a full-stack web application that analyzes chest and musculoskeletal (bone) X-ray images using deep learning. The platform provides AI-assisted image interpretation, Grad-CAM visual explanations, downloadable reports, and an intuitive web interface built for research and educational purposes.

Disclaimer: This project is intended for educational and research use only. It is not approved for clinical diagnosis or medical decision-making.

Features
Chest X-ray Analysis
Detects common thoracic abnormalities
Normal vs. abnormal assessment
Multi-label classification
Grad-CAM heatmap visualization
AI-generated interpretation
Downloadable PDF report
Bone X-ray Analysis
Musculoskeletal body-part classification
Abnormality detection
Confidence scores
Grad-CAM visualization
Structured report generation
Model Performance
Chest Model

Dataset

CheXpert

Architecture

DenseNet-121
Bone Models

Dataset

MURA (Stanford)

Body-Part Classifier

Validation Accuracy: 96.7%

Abnormality Classifier

Validation AUC: 0.871
Tech Stack
Frontend
Next.js
TypeScript
React
Tailwind CSS
Backend
FastAPI
Python
Deep Learning
PyTorch
Torchvision
Grad-CAM
Deployment
Vercel (Frontend)
Railway (Backend)
Repository Structure
frontend/
    Next.js application

backend/
    FastAPI API
    Chest model
    Bone model
    Report generation

ai/
    Chest models
    Bone models
    Validation metrics

models/
    Trained model weights
Example Workflow
Upload a chest or bone X-ray.
The image is validated.
The appropriate AI model performs inference.
Confidence scores and predictions are generated.
Grad-CAM highlights influential image regions.
Results are displayed.
A PDF report can be downloaded.
Research Datasets
CheXpert — Stanford University
MURA — Stanford University
Future Improvements
Additional imaging modalities
More anatomical regions
Faster inference
User authentication
Patient history support
Longitudinal study comparison
Cloud-based model optimization
Installation

Clone the repository:

git clone https://github.com/lrglazer/medVision_AI.git

Install frontend:

cd frontend
npm install
npm run dev

Install backend:

cd backend
pip install -r requirements.txt
uvicorn main:app --reload
Live Demo

🌐 Web App: https://med-vision-maoss267a-leya3.vercel.app/

💻 GitHub: https://github.com/lrglazer/medVision_AI

Author

Leya Glazer

Biomedical Engineering & Computer Science Student
The University of Texas at Dallas

License

This project is licensed under the MIT License.

Acknowledgments
Stanford ML Group for the CheXpert dataset
Stanford AIMI for the MURA dataset
PyTorch
FastAPI
Next.js
Vercel
Railway
