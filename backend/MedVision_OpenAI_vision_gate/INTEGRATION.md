# MedVision OpenAI vision gate integration

## 1. Install the OpenAI Python SDK

Run this in PowerShell from the MedVision root:

```powershell
& ".\ai\chest\.venv\Scripts\python.exe" -m pip install --upgrade openai pydantic
```

## 2. Configure the API key

For the current PowerShell window:

```powershell
$env:OPENAI_API_KEY="YOUR_API_KEY_HERE"
```

Do not put the key in frontend code and do not commit it to GitHub.

Optional model override:

```powershell
$env:OPENAI_VISION_GATE_MODEL="gpt-5.6"
```

## 3. Chest integration: backend/main.py

Add:

```python
from backend.openai_vision_gate import VisionGateError, get_vision_gate
```

Inside both:
- `/api/chest/predict`
- `/api/chest/report`

Immediately after:

```python
image = Image.open(io.BytesIO(await file.read()))
```

add:

```python
validation = get_vision_gate().classify(image, expected="chest")

if not validation.accepted:
    raise HTTPException(
        status_code=422,
        detail=(
            validation.reason
            + " Please upload a frontal or lateral chest radiograph."
        ),
    )
```

In each endpoint, make sure this appears before `except UnidentifiedImageError`:

```python
except HTTPException:
    raise
except VisionGateError as error:
    raise HTTPException(
        status_code=503,
        detail=(
            "The image-validation service is temporarily unavailable. "
            "The medical model was not run."
        ),
    ) from error
```

Optionally add to the successful JSON response:

```python
"input_validation": {
    "study_type": validation.study_type.value,
    "accepted": validation.accepted,
    "confidence": validation.confidence,
    "display_confidence": f"{validation.confidence:.1%}",
    "reason": validation.reason,
    "detected_region": validation.detected_region,
},
```

## 4. Bone integration: backend/bone_api.py

Add:

```python
from backend.openai_vision_gate import VisionGateError, get_vision_gate
```

Inside both:
- `/api/bone/predict`
- `/api/bone/report`

Immediately after:

```python
image = Image.open(io.BytesIO(await file.read()))
```

add:

```python
validation = get_vision_gate().classify(image, expected="bone")

if not validation.accepted:
    raise HTTPException(
        status_code=422,
        detail=(
            validation.reason
            + " Please upload an elbow, finger, forearm, hand, humerus, "
            "shoulder, or wrist radiograph."
        ),
    )
```

In each endpoint, make sure this appears before `except UnidentifiedImageError`:

```python
except HTTPException:
    raise
except VisionGateError as error:
    raise HTTPException(
        status_code=503,
        detail=(
            "The image-validation service is temporarily unavailable. "
            "The medical model was not run."
        ),
    ) from error
```

Optionally include the same `input_validation` object in the successful JSON response.

## 5. Start the backend with the key present

In the same PowerShell window:

```powershell
cd C:\Users\Owner\Documents\Medvision
$env:OPENAI_API_KEY="YOUR_API_KEY_HERE"
& ".\ai\chest\.venv\Scripts\python.exe" -m uvicorn backend.main:app --reload --port 8000
```

## 6. Test

Test both pages with:

- A real chest X-ray
- A real supported MURA bone X-ray
- A birthday-cake photograph
- A pet photograph
- A screenshot
- A chest X-ray uploaded to Bone Specialist
- A bone X-ray uploaded to Chest Specialist

Wrong and unsupported images should return HTTP 422 and never reach the DenseNet model.
