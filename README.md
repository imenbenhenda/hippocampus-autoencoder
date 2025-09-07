# Hippocampus MRI Anomaly Detection with 3D Autoencoder

This project implements a **3D convolutional autoencoder** in Python to detect anomalies in **hippocampus MRI scans**. The autoencoder learns to reconstruct normal MRI volumes, and anomalies are detected based on reconstruction errors.

---

## Features

- Preprocessing of 3D MRI volumes: normalization and resizing.
- 3D Convolutional Autoencoder for unsupervised anomaly detection.
- Automatic anomaly threshold computation based on normal MRI scans.
- Visualizations:
  - Slices in **sagittal, coronal, and axial planes** for original, reconstructed, and difference images.
  - Histogram of voxel-wise differences to highlight anomalies.
- Scores output for each volume: **mean** and **maximum reconstruction error**.
- Decision: **Normal** or **Anomaly detected** based on the threshold.

---

## Project Structure

Hippocampus_Project/
├─ data/
│ └─ Hippocampus_Dataset/
│ ├─ train/
│ └─ test/
│ └─ imagesTs/ # Test MRI volumes
├─ models/
│ └─ conv3d_autoencoder.pth # Pretrained model
├─ scripts/
│ ├─ anomaly_detection.py # Main script for anomaly detection
│ ├─ model.py # 3D autoencoder model
│ └─ preprocess.py # Normalization and resizing utilities
| |_train.py 
└─ README.md
## Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- NiBabel

Run the anomaly detection script:
python scripts/anomaly_detection.py


The script will:
Compute the anomaly threshold automatically from normal volumes.
Calculate mean and maximum reconstruction errors for each volume.
Display slice-wise visualizations.
Print the decision: ✅ Normal or ❌ Anomaly detected.

Example Output :
Seuil d'anomalie calculé automatiquement: 0.064310
--- Traitement de hippocampus_002.nii.gz ---
Score d’anomalie moyen: 0.045151
Score d’anomalie maximal: 0.703641
✅ Volume normal