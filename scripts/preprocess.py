import os                     
import nibabel as nib          
import numpy as np            
from scipy.ndimage import zoom  
import torch                  

# -----------------------------
# Parameters
# -----------------------------
DATA_PATH = "../data/Hippocampus_Dataset/train"  
TRAIN_IMAGES = "images/imagesTr"                 
TRAIN_LABELS = "labels/labelsTr"              

TARGET_SHAPE = (64, 64, 64)  
USE_HIPPOCAMPUS_ONLY = True   

# -----------------------------
# Functions
# -----------------------------

def normalize_volume(volume):
    """
    Normalize volume intensities between 0 and 1
    """
    vol = volume.astype(np.float32)  
    return (vol - vol.min()) / (vol.max() - vol.min() + 1e-8) 

def resize_volume(volume, target_shape=TARGET_SHAPE):
    """
    Resize volume to match target_shape size
    """
    # Calculate zoom factor for each dimension
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    return zoom(volume, factors, order=1) 

def crop_to_hippocampus(volume, mask):
    """
    Crop volume and mask to hippocampus region only
    """
    coords = np.array(np.nonzero(mask))  
    if coords.size == 0:  
        return volume, mask
    min_coords = coords.min(axis=1)  
    max_coords = coords.max(axis=1) + 1 
    
    return (volume[min_coords[0]:max_coords[0],
                   min_coords[1]:max_coords[1],
                   min_coords[2]:max_coords[2]],
            mask[min_coords[0]:max_coords[0],
                 min_coords[1]:max_coords[1],
                 min_coords[2]:max_coords[2]])

# -----------------------------
# Loading and preprocessing
# -----------------------------

image_files = sorted([f for f in os.listdir(os.path.join(DATA_PATH, TRAIN_IMAGES)) if f.endswith(".nii.gz")])
label_files = sorted([f for f in os.listdir(os.path.join(DATA_PATH, TRAIN_LABELS)) if f.endswith(".nii.gz")])

X_train = []  
Y_train = [] 


for img_file, lbl_file in zip(image_files, label_files):
    # Build full paths
    img_path = os.path.join(DATA_PATH, TRAIN_IMAGES, img_file)
    lbl_path = os.path.join(DATA_PATH, TRAIN_LABELS, lbl_file)
    
    # Load MRI volumes and masks
    img = nib.load(img_path).get_fdata()
    lbl = nib.load(lbl_path).get_fdata()
    
    # Normalize MRI intensities
    img = normalize_volume(img)
    
    # Crop to hippocampus if requested
    if USE_HIPPOCAMPUS_ONLY:
        img, lbl = crop_to_hippocampus(img, lbl)
    
    # Resize all volumes to TARGET_SHAPE
    img = resize_volume(img, TARGET_SHAPE)
    lbl = resize_volume(lbl, TARGET_SHAPE)
    
    # Add channel dimension 
    img = np.expand_dims(img, axis=0)
    lbl = np.expand_dims(lbl, axis=0)
    
    # Add preprocessed volumes to lists
    X_train.append(img)
    Y_train.append(lbl)

# Convert lists to PyTorch tensors
X_train_tensor = torch.tensor(np.stack(X_train), dtype=torch.float32)
Y_train_tensor = torch.tensor(np.stack(Y_train), dtype=torch.float32)

# Display final shape for verification
print("X_train shape:", X_train_tensor.shape)  # (nb_volumes, 1, 64, 64, 64)
print("Y_train shape:", Y_train_tensor.shape)
print("Preprocessing completed!")