import os
import nibabel as nib
import torch
import matplotlib.pyplot as plt
from model import Conv3DAutoencoder
import numpy as np

# -----------------------------
# Paramètres
# -----------------------------
DATA_PATH = "../data/Hippocampus_Dataset/test"
TEST_IMAGES = "imagesTs"  # dossier contenant les IRM de test
MODEL_PATH = "../models/conv3d_autoencoder.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Lister les fichiers de test
# -----------------------------
test_files = sorted([f for f in os.listdir(os.path.join(DATA_PATH, TEST_IMAGES)) if f.endswith(".nii.gz")])

# -----------------------------
# Charger le modèle
# -----------------------------
model = Conv3DAutoencoder().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -----------------------------
# Fonction pour détecter les anomalies
# -----------------------------
def detect_anomaly(img_path):
    # Charger l'image
    img = nib.load(img_path).get_fdata()
    
    # Normalisation
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    # Redimensionner si nécessaire (pour correspondre à TARGET_SHAPE=64)
    if img.shape != (64,64,64):
        from scipy.ndimage import zoom
        factors = [64 / s for s in img.shape]
        img = zoom(img, factors, order=1)
    
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,64,64,64)

    # Reconstruction
    with torch.no_grad():
        recon = model(img_tensor)
    
    recon = recon.squeeze().cpu().numpy()  # reconstruction en numpy

    # Calcul de la différence
    diff = np.abs(img - recon)

    # Retourner la différence moyenne comme score
    return diff, np.mean(diff)

# -----------------------------
# Tester tous les volumes
# -----------------------------
anomaly_scores = []

for f in test_files:
    path = os.path.join(DATA_PATH, TEST_IMAGES, f)
    diff, score = detect_anomaly(path)
    anomaly_scores.append((f, score))
    print(f"{f} | Score d’anomalie: {score:.5f}")

# Trier par score décroissant (les plus anormaux en premier)
anomaly_scores.sort(key=lambda x: x[1], reverse=True)

print("\nTop 5 volumes les plus anormaux :")
for f, score in anomaly_scores[:5]:
    print(f"{f} | Score: {score:.5f}")

# -----------------------------
# Affichage du volume le plus anormal
# -----------------------------
top_file = os.path.join(DATA_PATH, TEST_IMAGES, anomaly_scores[0][0])
diff, _ = detect_anomaly(top_file)
img = nib.load(top_file).get_fdata()
# Redimensionner si nécessaire
if img.shape != (64,64,64):
    from scipy.ndimage import zoom
    factors = [64 / s for s in img.shape]
    img = zoom(img, factors, order=1)

slice_z = 32  # coupe médiane

plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.title("IRM originale")
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
plt.title("Différence (anomalie)")
plt.imshow(diff[:,:,slice_z], cmap='hot')
plt.axis('off')

plt.show()
