import tkinter as tk
import threading
import pyautogui
import time
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader

# File to store mouse movement data
OUTPUT_FILE = "mouse_data.csv"
MODEL_FILE = "diffusion_model.pth"
SEQ_LENGTH = 50


# Define a simple neural network for smoothing
class DiffusionModel(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        """Initialize the neural network model."""
        super(DiffusionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Hidden layer
        self.fc3 = nn.Linear(hidden_dim, input_dim)  # Output layer
        self.relu = nn.ReLU()  # Activation function

    def forward(self, x):
        """Forward pass of the model."""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load and preprocess mouse movement data
def load_mouse_data(filename):
    """Load and preprocess the mouse movement data from CSV file."""
    df = pd.read_csv(filename)
    df['timestamp'] -= df['timestamp'].min()  # Normalize timestamps
    df[['x', 'y']] = df[['x', 'y']].astype(float)  # Ensure float type
    return df


class MouseDataset(Dataset):
    """Dataset class for handling sequences of mouse movements."""

    def __init__(self, data, seq_length=SEQ_LENGTH):
        self.data = data
        self.seq_length = seq_length  # Sequence length

    def __len__(self):
        return len(self.data) - self.seq_length  # Total available sequences

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx:idx + self.seq_length, 1:], dtype=torch.float32),
                torch.tensor(self.data[idx:idx + self.seq_length, 1:], dtype=torch.float32))


# Function to collect mouse movement data
def collect_data():
    """Collect mouse movement data and store it in a CSV file."""
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "x", "y"])

    try:
        print("Collecting mouse movement data...")
        while collecting:
            x, y = pyautogui.position()
            timestamp = time.time()
            with open(OUTPUT_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, x, y])
            time.sleep(0.1)  # Data collection frequency
    except KeyboardInterrupt:
        print("Data collection stopped.")


# Train the model
def train_model():
    """Train the neural network on the collected mouse movement data."""
    df = load_mouse_data(OUTPUT_FILE)
    dataset = MouseDataset(df.to_numpy())
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = DiffusionModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(20):
        for noisy, clean in dataloader:
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), MODEL_FILE)
    print("Model training complete.")


# Real-time mouse smoothing
def smooth_mouse_movement():
    """Smooth real-time mouse movement using the trained model."""
    model = DiffusionModel()
    model.load_state_dict(torch.load(MODEL_FILE))
    model.eval()

    buffer = []  # Stores recent mouse positions
    global smoothing
    smoothing = True

    try:
        while smoothing:
            x, y = pyautogui.position()
            buffer.append([x, y])
            if len(buffer) > SEQ_LENGTH:
                buffer.pop(0)
                input_data = torch.tensor([buffer], dtype=torch.float32)
                smoothed_position = model(input_data).detach().numpy()[0, -1]
                pyautogui.moveTo(int(smoothed_position[0]), int(smoothed_position[1]), duration=0.01)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Real-time smoothing stopped.")


# GUI setup
def start_learning():
    """Start the data collection process."""
    global collecting
    collecting = True
    threading.Thread(target=collect_data, daemon=True).start()


def stop_learning():
    """Stop data collection and train the model."""
    global collecting
    collecting = False
    train_model()


def start_smoothing():
    """Start real-time mouse smoothing."""
    global smoothing
    smoothing = True
    threading.Thread(target=smooth_mouse_movement, daemon=True).start()


def stop_smoothing():
    """Stop the real-time smoothing process."""
    global smoothing
    smoothing = False


# Creating the GUI window
tk_root = tk.Tk()
tk_root.title("Mouse Movement Smoother")

# Adding buttons to the GUI
tk.Button(tk_root, text="Start Learning", command=start_learning).pack(pady=10)
tk.Button(tk_root, text="Stop Learning", command=stop_learning).pack(pady=10)
tk.Button(tk_root, text="Start Smoothing", command=start_smoothing).pack(pady=10)
tk.Button(tk_root, text="Stop Smoothing", command=stop_smoothing).pack(pady=10)

# Start the Tkinter event loop
tk_root.mainloop()
