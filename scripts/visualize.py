import os              
import nibabel as nib   
import matplotlib.pyplot as plt 

# -----------------------------
# Parameters
# -----------------------------
DATA_PATH = "../data/Hippocampus_Dataset/train"  
TRAIN_IMAGES = "images/imagesTr"               
TRAIN_LABELS = "labels/labelsTr"              

# -----------------------------
# List files
# -----------------------------
image_files = sorted([f for f in os.listdir(os.path.join(DATA_PATH, TRAIN_IMAGES)) if f.endswith(".nii.gz")])
label_files = sorted([f for f in os.listdir(os.path.join(DATA_PATH, TRAIN_LABELS)) if f.endswith(".nii.gz")])

# -----------------------------
# Load an example
# -----------------------------
img_path = os.path.join(DATA_PATH, TRAIN_IMAGES, image_files[0])
lbl_path = os.path.join(DATA_PATH, TRAIN_LABELS, label_files[0])
img = nib.load(img_path).get_fdata()  
lbl = nib.load(lbl_path).get_fdata()  

# -----------------------------
# 2D display
# -----------------------------

plt.figure(figsize=(12,5))  
plt.subplot(1,2,1)         
plt.title("Hippocampus MRI")
plt.imshow(img[:,:,slice_z], cmap='gray')  
plt.axis('off')          

plt.subplot(1,2,2)        
plt.title("Hippocampus mask")
plt.imshow(lbl[:,:,slice_z], cmap='gray')  
plt.axis('off')

plt.show() 

# -----------------------------
# MRI + mask overlay
# -----------------------------
plt.figure(figsize=(6,6))  
plt.title("Mask overlay on MRI")
plt.imshow(img[:,:,slice_z], cmap='gray')       
plt.imshow(lbl[:,:,slice_z], cmap='Reds', alpha=0.4) 
plt.axis('off')

plt.show() 