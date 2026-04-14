import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from app.core.model_utils import get_model
from app.core.image_utils import load_dataset
import os

def train_model(dataset_name="fashion", model_type="cnn_vae", epochs=5):
    print(f"Iniciando treinamento rápido de {model_type} no dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, train=True)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_type, latent_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion_mse = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            if model_type == "cnn_vae":
                recon, mu, logvar = model(data)
                mse_loss = criterion_mse(recon, data)
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                kld_loss /= data.size(0) * 28 * 28
                loss = mse_loss + 0.005 * kld_loss
            else:
                recon = model(data)
                loss = criterion_mse(recon, data)
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Batch {batch_idx}/{len(loader)} - Loss: {loss.item():.4f}")
                
        print(f"=== Epoch {epoch+1} concluída. Loss Média: {total_loss/len(loader):.4f} ===")
        
    save_path = f"app/core/{dataset_name}_{model_type}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Pesos salvos em {save_path} com sucesso.")

if __name__ == "__main__":
    train_model("cifar10", "cnn_vae", epochs=5)
    train_model("fashion", "cnn_vae", epochs=5)
    train_model("fashion", "cnn_ae", epochs=5)
    train_model("mnist", "cnn_vae", epochs=3)
    train_model("mnist", "cnn_ae", epochs=3)
