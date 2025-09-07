import os               # pour gérer les chemins et lister les fichiers dans les dossiers
import nibabel as nib   # pour lire les fichiers NIfTI (.nii.gz) contenant les volumes IRM
import matplotlib.pyplot as plt  # pour afficher les images 2D et superpositions

# -----------------------------
# Paramètres
# -----------------------------
DATA_PATH = "../data/Hippocampus_Dataset/train"  # chemin vers le dossier "train"
TRAIN_IMAGES = "images/imagesTr"                # sous-dossier contenant les IRM d’entraînement
TRAIN_LABELS = "labels/labelsTr"               # sous-dossier contenant les masques hippocampe

# -----------------------------
# Lister les fichiers
# -----------------------------
# On récupère uniquement les fichiers .nii.gz pour éviter les fichiers cachés ou autres types
image_files = sorted([f for f in os.listdir(os.path.join(DATA_PATH, TRAIN_IMAGES)) if f.endswith(".nii.gz")])
label_files = sorted([f for f in os.listdir(os.path.join(DATA_PATH, TRAIN_LABELS)) if f.endswith(".nii.gz")])
# sorted() pour que les images et masques soient bien alignés dans le même ordre

# -----------------------------
# Charger un exemple
# -----------------------------
# On prend le premier fichier pour visualiser un exemple
img_path = os.path.join(DATA_PATH, TRAIN_IMAGES, image_files[0])
lbl_path = os.path.join(DATA_PATH, TRAIN_LABELS, label_files[0])

# Lecture des volumes avec nibabel et conversion en array NumPy
img = nib.load(img_path).get_fdata()  # IRM
lbl = nib.load(lbl_path).get_fdata()  # Masque hippocampe

# -----------------------------
# Affichage 2D
# -----------------------------
slice_z = img.shape[2] // 2  # coupe médiane dans l’axe Z (axial)

plt.figure(figsize=(12,5))  # taille de la figure

# Affichage de l’IRM
plt.subplot(1,2,1)          # 1 ligne, 2 colonnes, 1er subplot
plt.title("IRM hippocampe")
plt.imshow(img[:,:,slice_z], cmap='gray')  # affiche la coupe médiane en niveaux de gris
plt.axis('off')            # on masque les axes pour plus de clarté

# Affichage du masque
plt.subplot(1,2,2)          # 1 ligne, 2 colonnes, 2ème subplot
plt.title("Masque hippocampe")
plt.imshow(lbl[:,:,slice_z], cmap='gray')  # affiche la coupe du masque en gris
plt.axis('off')

plt.show()  # affichage de la figure avec les deux images côte à côte

# -----------------------------
# Superposition IRM + masque
# -----------------------------
plt.figure(figsize=(6,6))  # figure carrée
plt.title("Superposition masque sur IRM")
plt.imshow(img[:,:,slice_z], cmap='gray')        # affiche l’IRM en arrière-plan
plt.imshow(lbl[:,:,slice_z], cmap='Reds', alpha=0.4)  # masque en rouge transparent par-dessus
plt.axis('off')

plt.show()  # affichage de la superposition
