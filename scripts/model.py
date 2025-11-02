import torch
import torch.nn as nn

# -----------------------------
# Model definition
# -----------------------------
class Conv3DAutoencoder(nn.Module):
    def __init__(self):
        super(Conv3DAutoencoder, self).__init__()
        
        # -----------------------------
        # Encoder
        # -----------------------------
        # Reduces volume dimensions while increasing channel count
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
        # Decoder
        # -----------------------------
        # Reconstructs original volume
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2), # -> (32,16,16,16)
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2), # -> (16,32,32,32)
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=2, stride=2),  # -> (1,64,64,64)
            nn.Sigmoid()  #
        )
        
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# -----------------------------
# Quick model test
# -----------------------------
if __name__ == "__main__":
    model = Conv3DAutoencoder()
    print("Conv3DAutoencoder model:")
    print(model)
    
    x = torch.randn((1,1,64,64,64))
    out = model(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    
    criterion = nn.MSELoss()
    loss = criterion(out, x)
    print("Example loss:", loss.item())