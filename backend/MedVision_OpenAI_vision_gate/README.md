# MedVision OpenAI vision gate

This package adds a vision-language-model routing gate before the Chest and
Bone DenseNet models.

It uses the OpenAI Responses API with image input and a Pydantic structured
output.

The gate accepts only:
- frontal or lateral chest radiographs for Chest Specialist
- elbow, finger, forearm, hand, humerus, shoulder, or wrist radiographs for
  Bone Specialist

The implementation rejects uncertain inputs and requires confidence >= 0.80.

## Important limitations

- This adds API cost and requires internet access.
- It should not be described as a clinical safety system.
- Do not send identifiable patient images unless your data handling,
  consent, contracts, and compliance setup permit it.
- For a public portfolio demo, use de-identified or public research images.
- The DenseNet medical model is not run when validation fails or when the
  validation service is unavailable.

Read `INTEGRATION.md` for exact installation steps.
