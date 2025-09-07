"""
model.py
Définition d'un autoencodeur 3D convolutionnel (CAE) pour volumes IRM hippocampe.
- Encodeur : compressions 3D Conv + MaxPooling
- Décodeur : reconstruction 3D ConvTranspose
- Fonction de perte : MSE
"""

import torch
import torch.nn as nn

# -----------------------------
# Définition du modèle
# -----------------------------
class Conv3DAutoencoder(nn.Module):
    def __init__(self):
        super(Conv3DAutoencoder, self).__init__()
        
        # -----------------------------
        # Encodeur
        # -----------------------------
        # Réduit les dimensions du volume tout en augmentant le nombre de canaux
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),  # (1,64,64,64) -> (16,64,64,64)
            nn.ReLU(),
            nn.MaxPool3d(2, 2),                          # -> (16,32,32,32)
            
            nn.Conv3d(16, 32, kernel_size=3, padding=1), # -> (32,32,32,32)
            nn.ReLU(),
            nn.MaxPool3d(2, 2),                          # -> (32,16,16,16)
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1), # -> (64,16,16,16)
            nn.ReLU(),
            nn.MaxPool3d(2, 2)                           # -> (64,8,8,8)
        )
        
        # -----------------------------
        # Décodeur
        # -----------------------------
        # Reconstruit le volume original
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2), # -> (32,16,16,16)
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2), # -> (16,32,32,32)
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=2, stride=2),  # -> (1,64,64,64)
            nn.Sigmoid()  # sortie entre 0 et 1 (comme normalisation)
        )
        
    def forward(self, x):
        # Passage dans l'encodeur puis le décodeur
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# -----------------------------
# Test rapide du modèle
# -----------------------------
if __name__ == "__main__":
    # Initialisation du modèle
    model = Conv3DAutoencoder()
    print("Modèle Conv3DAutoencoder:")
    print(model)
    
    # Exemple d'entrée : batch_size=1, canal=1, dimensions 64x64x64
    x = torch.randn((1,1,64,64,64))
    out = model(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    
    # Définir la fonction de perte (MSE)
    criterion = nn.MSELoss()
    loss = criterion(out, x)
    print("Exemple de perte:", loss.item())
