import os
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from model import Conv3DAutoencoder
import numpy as np

# -----------------------------
# Parameters
# -----------------------------
DATA_PATH = "../data/Hippocampus_Dataset/test"
TEST_IMAGES = "imagesTs"  
MODEL_PATH = "../models/conv3d_autoencoder.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# List test files
# -----------------------------
test_files = sorted([f for f in os.listdir(os.path.join(DATA_PATH, TEST_IMAGES)) if f.endswith(".nii.gz")])

# -----------------------------
# Load model
# -----------------------------
model = Conv3DAutoencoder().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -----------------------------
# Function to detect anomalies
# -----------------------------
def detect_anomaly(img_path):
  
    img = nib.load(img_path).get_fdata()
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    if img.shape != (64,64,64):
        from scipy.ndimage import zoom
        factors = [64 / s for s in img.shape]
        img = zoom(img, factors, order=1)
    
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,64,64,64)

    with torch.no_grad():
        recon = model(img_tensor)
    
    recon = recon.squeeze().cpu().numpy() 

    diff = np.abs(img - recon)

    return diff, np.mean(diff)

# -----------------------------
# Test all volumes
# -----------------------------
anomaly_scores = []

for f in test_files:
    path = os.path.join(DATA_PATH, TEST_IMAGES, f)
    diff, score = detect_anomaly(path)
    anomaly_scores.append((f, score))
    print(f"{f} | Anomaly score: {score:.5f}")

anomaly_scores.sort(key=lambda x: x[1], reverse=True)

print("\nTop 5 most abnormal volumes:")
for f, score in anomaly_scores[:5]:
    print(f"{f} | Score: {score:.5f}")

# -----------------------------
# Display most abnormal volume
# -----------------------------
top_file = os.path.join(DATA_PATH, TEST_IMAGES, anomaly_scores[0][0])
diff, _ = detect_anomaly(top_file)
img = nib.load(top_file).get_fdata()
# Resize if necessary
if img.shape != (64,64,64):
    from scipy.ndimage import zoom
    factors = [64 / s for s in img.shape]
    img = zoom(img, factors, order=1)

slice_z = 32  

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("Original MRI")
plt.imshow(img[:,:,slice_z], cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Reconstruction")
recon_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    recon = model(recon_tensor).squeeze().cpu().numpy()
plt.imshow(recon[:,:,slice_z], cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Difference (anomaly)")
plt.imshow(diff[:,:,slice_z], cmap='hot')
plt.axis('off')

plt.show()