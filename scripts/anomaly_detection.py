import os
import nibabel as nib
import torch
import matplotlib.pyplot as plt
import numpy as np
from model import Conv3DAutoencoder
import preprocess  # pour normalize_volume et resize_volume

# -----------------------------
# Paramètres
# -----------------------------
DATA_PATH = "../data/Hippocampus_Dataset/test"
NORMAL_IMAGES = "normal"  # sous-dossier contenant uniquement des IRM normales pour calcul du seuil
TEST_IMAGES = "imagesTs"
MODEL_PATH = "../models/conv3d_autoencoder.pth"
TARGET_SHAPE = (64, 64, 64)

# -----------------------------
# Charger le modèle
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Conv3DAutoencoder().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -----------------------------
# Fonctions d'affichage amélioré
# -----------------------------
def show_slices(img, recon, diff):
    slice_indices = [TARGET_SHAPE[0]//2, TARGET_SHAPE[1]//2, TARGET_SHAPE[2]//2]
    planes = ['Sagittal', 'Coronal', 'Axial']

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for i, s in enumerate(slice_indices):
        axes[i,0].imshow(img[s,:,:] if i==0 else img[:,s,:] if i==1 else img[:,:,s], cmap='gray')
        axes[i,0].set_title(f"IRM {planes[i]}")
        axes[i,0].axis('off')

        axes[i,1].imshow(recon[s,:,:] if i==0 else recon[:,s,:] if i==1 else recon[:,:,s], cmap='gray')
        axes[i,1].set_title(f"Recon {planes[i]}")
        axes[i,1].axis('off')

        axes[i,2].imshow(diff[s,:,:] if i==0 else diff[:,s,:] if i==1 else diff[:,:,s], cmap='hot')
        axes[i,2].set_title(f"Diff {planes[i]}")
        axes[i,2].axis('off')

    plt.tight_layout()
    plt.show()

def show_diff_hist(diff):
    plt.figure(figsize=(6,4))
    plt.hist(diff.flatten(), bins=50, color='orange')
    plt.title("Distribution des valeurs de différence")
    plt.xlabel("Valeur de différence")
    plt.ylabel("Nombre de voxels")
    plt.show()

# -----------------------------
# Détection d'anomalie pour un volume
# -----------------------------
def detect_anomaly(img_path, threshold):
    img = nib.load(img_path).get_fdata()
    img = preprocess.normalize_volume(img)
    img = preprocess.resize_volume(img, TARGET_SHAPE)
    
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        recon = model(img_tensor)
    recon = recon.squeeze().cpu().numpy()
    
    diff = np.abs(img - recon)
    mean_score = np.mean(diff)
    max_score = np.max(diff)

    # Affichage scores
    print(f"Score d’anomalie moyen: {mean_score:.6f}")
    print(f"Score d’anomalie maximal: {max_score:.6f}")

    # Décision Normal / Anomalie
    if mean_score > threshold:
        print("❌ Anomalie détectée !")
    else:
        print("✅ Volume normal")

    # Affichage amélioré
    show_slices(img, recon, diff)
    show_diff_hist(diff)

    return mean_score, max_score

# -----------------------------
# Calcul du seuil automatique à partir des volumes normaux
# -----------------------------
def compute_threshold():
    # Utiliser le dossier des tests pour calculer le seuil
    normal_path = os.path.join(DATA_PATH, TEST_IMAGES)  
    normal_files = sorted([f for f in os.listdir(normal_path) if f.endswith(".nii.gz")])
    scores = []
    for f in normal_files:
        img_path = os.path.join(normal_path, f)
        img = nib.load(img_path).get_fdata()
        img = preprocess.normalize_volume(img)
        img = preprocess.resize_volume(img, TARGET_SHAPE)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model(img_tensor)
        recon = recon.squeeze().cpu().numpy()
        diff = np.abs(img - recon)
        scores.append(np.mean(diff))
    scores = np.array(scores)
    threshold = np.mean(scores) + 2 * np.std(scores)
    print(f"Seuil d'anomalie calculé automatiquement: {threshold:.6f}")
    return threshold


# -----------------------------
# Exemple sur tous les volumes de test
# -----------------------------
if __name__ == "__main__":
    threshold = compute_threshold()
    test_path_full = os.path.join(DATA_PATH, TEST_IMAGES)
    test_files = sorted([f for f in os.listdir(test_path_full) if f.endswith(".nii.gz")])
    
    for test_file in test_files:
        print(f"\n--- Traitement de {test_file} ---")
        test_path = os.path.join(test_path_full, test_file)
        detect_anomaly(test_path, threshold)
