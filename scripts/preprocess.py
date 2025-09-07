"""
preprocess.py
Script pour prétraiter les volumes IRM hippocampe pour l'autoencodeur 3D
- Normalisation des intensités
- Resize / recadrage à une dimension fixe
- (Optionnel) Extraction de l'hippocampe via le masque
- Création de tensors PyTorch X_train et Y_train
"""

import os                     # Pour naviguer dans les dossiers et fichiers
import nibabel as nib          # Pour lire les fichiers IRM au format .nii.gz
import numpy as np            # Pour manipuler les tableaux / volumes 3D
from scipy.ndimage import zoom  # Pour redimensionner les volumes 3D
import torch                  # Pour créer les tensors PyTorch pour l'entraînement

# -----------------------------
# Paramètres
# -----------------------------
DATA_PATH = "../data/Hippocampus_Dataset/train"  # chemin vers le dossier train
TRAIN_IMAGES = "images/imagesTr"                 # sous-dossier contenant les IRM
TRAIN_LABELS = "labels/labelsTr"                # sous-dossier contenant les masques

TARGET_SHAPE = (64, 64, 64)  # taille cible des volumes pour le modèle
USE_HIPPOCAMPUS_ONLY = True   # True si on veut ne garder que l'hippocampe

# -----------------------------
# Fonctions
# -----------------------------

def normalize_volume(volume):
    """
    Normalisation des intensités du volume entre 0 et 1
    """
    vol = volume.astype(np.float32)  # convertir en float32 pour plus de précision
    return (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)  # +1e-8 pour éviter division par zéro

def resize_volume(volume, target_shape=TARGET_SHAPE):
    """
    Redimensionner le volume pour qu'il ait la même taille que target_shape
    """
    # Calcul du facteur de zoom pour chaque dimension
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1)  # interpolation linéaire pour redimensionner

def crop_to_hippocampus(volume, mask):
    """
    Recadrer le volume et le masque sur la région de l'hippocampe uniquement
    """
    coords = np.array(np.nonzero(mask))  # positions des voxels où le masque != 0
    if coords.size == 0:  # si le masque est vide, retourner le volume entier
        return volume, mask
    min_coords = coords.min(axis=1)  # coordonnée min de l'hippocampe
    max_coords = coords.max(axis=1) + 1  # coordonnée max (+1 pour inclure le dernier voxel)
    # Retourne le volume et le masque recadrés
    return (volume[min_coords[0]:max_coords[0],
                   min_coords[1]:max_coords[1],
                   min_coords[2]:max_coords[2]],
            mask[min_coords[0]:max_coords[0],
                 min_coords[1]:max_coords[1],
                 min_coords[2]:max_coords[2]])

# -----------------------------
# Chargement et prétraitement
# -----------------------------

# Lister uniquement les fichiers .nii.gz pour éviter les fichiers cachés
image_files = sorted([f for f in os.listdir(os.path.join(DATA_PATH, TRAIN_IMAGES)) if f.endswith(".nii.gz")])
label_files = sorted([f for f in os.listdir(os.path.join(DATA_PATH, TRAIN_LABELS)) if f.endswith(".nii.gz")])

# Listes pour stocker tous les volumes prétraités
X_train = []  # IRM
Y_train = []  # masques correspondants

# Boucle sur chaque paire image/masque
for img_file, lbl_file in zip(image_files, label_files):
    # Construire les chemins complets
    img_path = os.path.join(DATA_PATH, TRAIN_IMAGES, img_file)
    lbl_path = os.path.join(DATA_PATH, TRAIN_LABELS, lbl_file)
    
    # Charger les volumes IRM et masques
    img = nib.load(img_path).get_fdata()
    lbl = nib.load(lbl_path).get_fdata()
    
    # Normaliser les intensités de l'IRM
    img = normalize_volume(img)
    
    # Recadrer sur l'hippocampe si demandé
    if USE_HIPPOCAMPUS_ONLY:
        img, lbl = crop_to_hippocampus(img, lbl)
    
    # Redimensionner pour tous les volumes à TARGET_SHAPE
    img = resize_volume(img, TARGET_SHAPE)
    lbl = resize_volume(lbl, TARGET_SHAPE)
    
    # Ajouter la dimension canal (PyTorch attend [canal, x, y, z])
    img = np.expand_dims(img, axis=0)
    lbl = np.expand_dims(lbl, axis=0)
    
    # Ajouter les volumes prétraités aux listes
    X_train.append(img)
    Y_train.append(lbl)

# Convertir les listes en tensors PyTorch
X_train_tensor = torch.tensor(np.stack(X_train), dtype=torch.float32)
Y_train_tensor = torch.tensor(np.stack(Y_train), dtype=torch.float32)

# Afficher la forme finale pour vérification
print("X_train shape:", X_train_tensor.shape)  # (nb_volumes, 1, 64, 64, 64)
print("Y_train shape:", Y_train_tensor.shape)
print("Prétraitement terminé !")
