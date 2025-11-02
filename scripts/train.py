import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from model import Conv3DAutoencoder  
import preprocess  

# -----------------------------
# Parameters
# -----------------------------
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-3
MODEL_DIR = "../models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Prepare data
# -----------------------------

X_train_tensor = preprocess.X_train_tensor  
dataset = TensorDataset(X_train_tensor, X_train_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# Initialize model
# -----------------------------
model = Conv3DAutoencoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# -----------------------------
# Training
# -----------------------------
for epoch in range(1, EPOCHS+1):
    model.train()
    train_loss = 0
    for batch in train_loader:
        inputs, _ = batch
        inputs = inputs.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
    
    train_loss /= train_size
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, _ = batch
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= val_size
    
    print(f"Epoch {epoch}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

# -----------------------------
# Save model
# -----------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
model_path = os.path.join(MODEL_DIR, "conv3d_autoencoder.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")