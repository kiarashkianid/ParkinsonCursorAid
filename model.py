import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pyautogui
import time
from torch.utils.data import Dataset, DataLoader

# Load and preprocess the data
def load_mouse_data(filename):
    df = pd.read_csv(filename)
    df['timestamp'] -= df['timestamp'].min()  # Normalize time
    df[['x', 'y']] = df[['x', 'y']].astype(float)
    return df

# Define a custom dataset for handling mouse movement sequences
class MouseDataset(Dataset):
    def __init__(self, data, seq_length=50):
        self.data = data
        self.seq_length = seq_length  # Length of sequences to be used as input
    
    def __len__(self):
        return len(self.data) - self.seq_length  # Ensure sequence length is valid
    
    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx:idx+self.seq_length, 1:], dtype=torch.float32), 
                torch.tensor(self.data[idx:idx+self.seq_length, 1:], dtype=torch.float32))

# Define a simple neural network for diffusion-based noise reduction
class DiffusionModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super(DiffusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Hidden layer
        self.fc3 = nn.Linear(hidden_dim, input_dim)  # Output layer
        self.relu = nn.ReLU()  # Activation function
    
    def forward(self, x):
        x = self.relu(self.fc1(x))  # Apply ReLU activation
        x = self.relu(self.fc2(x))  # Apply ReLU activation again
        x = self.fc3(x)  # Generate output
        return x

# Training function for the model
def train_model(data_file, epochs=20, batch_size=32, learning_rate=0.001):
    df = load_mouse_data(data_file)  # Load mouse movement data
    dataset = MouseDataset(df.to_numpy())  # Create dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # DataLoader for batching
    
    model = DiffusionModel()  # Initialize model
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimizer
    criterion = nn.MSELoss()  # Mean squared error loss function
    
    for epoch in range(epochs):
        for noisy, clean in dataloader:
            optimizer.zero_grad()  # Clear previous gradients
            output = model(noisy)  # Forward pass
            loss = criterion(output, clean)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")  # Print loss per epoch
    
    torch.save(model.state_dict(), "diffusion_model.pth")  # Save trained model
    print("Model training complete.")

# Real-time smoothing function for mouse movements
def smooth_mouse_movement():
    model = DiffusionModel()  # Initialize model
    model.load_state_dict(torch.load("diffusion_model.pth"))  # Load trained model
    model.eval()  # Set model to evaluation mode
    
    buffer = []  # Buffer to store recent mouse positions
    seq_length = 50  # Sequence length used for smoothing
    
    try:
        while True:
            x, y = pyautogui.position()  # Capture current mouse position
            buffer.append([x, y])  # Append new position to buffer
            
            if len(buffer) > seq_length:
                buffer.pop(0)  # Maintain fixed sequence length
                input_data = torch.tensor([buffer], dtype=torch.float32)  # Convert buffer to tensor
                smoothed_position = model(input_data).detach().numpy()[0, -1]  # Predict smoothed position
                pyautogui.moveTo(int(smoothed_position[0]), int(smoothed_position[1]), duration=0.01)  # Move mouse
            
            time.sleep(0.01)  # Small delay to avoid excessive CPU usage
    except KeyboardInterrupt:
        print("Real-time smoothing stopped.")  # Stop loop when interrupted

if __name__ == "__main__":
    smooth_mouse_movement()  # Start real-time smoothing
