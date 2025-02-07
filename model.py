import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# Load and preprocess the data
def load_mouse_data(filename):
    df = pd.read_csv(filename)
    df['timestamp'] -= df['timestamp'].min()  # Normalize time
    df[['x', 'y']] = df[['x', 'y']].astype(float)
    return df

class MouseDataset(Dataset):
    def __init__(self, data, seq_length=50):
        self.data = data
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx:idx+self.seq_length, 1:], dtype=torch.float32), 
                torch.tensor(self.data[idx:idx+self.seq_length, 1:], dtype=torch.float32))

# Define a simple diffusion model
class DiffusionModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super(DiffusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Training loop
def train_model(data_file, epochs=20, batch_size=32, learning_rate=0.001):
    df = load_mouse_data(data_file)
    dataset = MouseDataset(df.to_numpy())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = DiffusionModel()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        for noisy, clean in dataloader:
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), "diffusion_model.pth")
    print("Model training complete.")

if __name__ == "__main__":
    train_model("mouse_data.csv")
