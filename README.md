# Industrial_Anomaly_Detection-on-MVtec
This repository contains an implementation of PatchCore for unsupervised anomaly detection on industrial components using the MVTec Anomaly Detection dataset. The approach utilizes a pretrained ResNet-50 backbone to extract features from images and compare them to a memory bank of normal samples.

The system supports both binary classification (good vs defective) and fine-grained defect localization.

---

## Features

- Patch-based feature extraction using ResNet-50 (Layer2 + Layer3)
- Memory bank construction from training (good) images
- Nearest-neighbor distance-based anomaly scoring
- Image-level and pixel-level anomaly detection
- Thresholding using both statistical and F1-score optimization
- Visual outputs: anomaly heatmaps and segmentation maps
- Evaluation with ROC-AUC, F1-score, and confusion matrix

---

## Dataset

**MVTec Anomaly Detection Dataset**

- 15 object categories (e.g., bottle, screw, metal_nut)
- Structure:
mvtec_anomaly_detection/
└── metal_nut/
├── train/
│ └── good/
└── test/
├── bent/
├── color/
├── flip/
├── scratch/
└── good/
