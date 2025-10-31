===========================================================
      ROPE CONDITION ANALYZER – OFFLINE AI PROJECT
===========================================================

PROJECT TITLE:
-----------------
"Elevator Rope Health Monitoring using AI and Computer Vision"

PROJECT OVERVIEW:
-----------------
This system analyzes images of elevator ropes to identify
and highlight possible degradation areas such as:
  1. Rust formation between strands
  2. Wire cuts (single or multiple)
  3. Low lubrication regions (planned future extension)

It uses a deep learning segmentation model (U-Net)
combined with OpenCV overlays for clear visualization.

===========================================================

PROJECT OBJECTIVE:
-----------------
The main goal is to automate the visual inspection process
for elevator maintenance teams to detect rope health issues
early and prevent accidents or breakdowns.

This solution can be deployed locally (offline) to ensure
data privacy for internal maintenance images.

===========================================================

TOOLS AND TECHNOLOGIES USED:
-----------------
1. Python 3.10
2. PyTorch (for model training)
3. OpenCV (for image processing)
4. LabelMe (for manual annotation)
5. Streamlit (for the user interface)
6. NumPy, Pillow, TorchVision

===========================================================

DATASET DESCRIPTION:
-----------------
Total Images: ~318 (internal KONE rope images)
Image Type: JPG
Image Resolution: Standardized to 256x256
Annotations: Created using LabelMe tool
Classes:
  - Background (0)
  - Rust (1)
  - Wire Cut (2)

===========================================================

TRAINING SUMMARY:
-----------------
Model: U-Net (custom lightweight)
Loss Function: Cross Entropy
Optimizer: Adam (lr = 0.001)
Epochs: 10
Batch Size: 2
Device: CPU / GPU supported

Trained Model Saved As:
  → models/unet_model.pth

===========================================================

APPLICATION WORKFLOW:
-----------------
1. Launch the app locally:
   > streamlit run app.py

2. Upload rope images (.jpg/.png)

3. Model predicts and highlights:
   - 🔴 Red → Rust zone
   - 🟢 Green → Wire cut zone

4. The app provides a condition summary:
   ✅ Healthy Rope
   🟡 Minor Rust
   🟠 Rust – Maintenance Needed
   ⚠️ Wire Cut – Replace Immediately

===========================================================

DATA SECURITY NOTE:
-----------------
- The entire project runs **offline**
- No data is sent or uploaded online
- Ideal for secure industrial demo environments

===========================================================

FUTURE ENHANCEMENTS:
-----------------
1. Include lubricant level classification
2. Integrate with real-time camera feed
3. Combine with rope tension & vibration KPIs
4. Deploy as part of KONE Predictive Maintenance Platform

===========================================================

PROJECT DEVELOPED BY:
-----------------
Name: PV
Role: Engineering Student (R&D Project Contributor)
Community: Digital Dreamers Den (D3)

Mentor: [Your mentor’s name here]
Company: KONE (Industrial AI Initiative)
Date: [Add date of demo or submission]

===========================================================
