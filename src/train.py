# File: src/__init__.py


import argparse
import torch
from torch.utils.data import DataLoader
from .data_loader import CFRPDataset
import torch.nn as nn
import torch.optim as optim

class SimpleCNN(nn.Module):
    def __init__(self, window_size):
        super(SimpleCNN, self).__init__()
        # One-dimensional conv expects (batch, channels, length)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        # After one pooling: length // 2
        fc_input_dim = (window_size // 2) * 16
        self.fc1 = nn.Linear(fc_input_dim, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch, length)
        x = x.unsqueeze(1)               # -> (batch, 1, length)
        x = self.conv1(x)                # -> (batch, 16, length)
        x = self.relu(x)
        x = self.pool(x)                 # -> (batch, 16, length//2)
        x = self.flatten(x)              # -> (batch, 16*(length//2))
        x = self.relu(self.fc1(x))       # -> (batch, 64)
        x = self.fc2(x)                  # -> (batch, 1)
        return x.squeeze(1)

def train(args):
    # Prepare dataset and dataloader
    dataset = CFRPDataset(
        root_dir=args.data_dir,
        layup=args.layup,
        window_size=args.window_size
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # Initialize model, loss, optimizer
    model = SimpleCNN(window_size=args.window_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for signals, cycles in dataloader:
            signals = signals.to(device)
            cycles = cycles.to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, cycles)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * signals.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch}/{args.epochs}] Loss: {epoch_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), args.output_path)
    print(f"Model saved to {args.output_path}")

if __name__ == "__main__":
    # To execute: python -m src.train --data-dir data --layup layup1 ...
    parser = argparse.ArgumentParser(description="Train Simple CNN on CFRP fatigue data")
    parser.add_argument("--data-dir", type=str, default="data", help="Root data directory")
    parser.add_argument("--layup", type=str, default="layup1", choices=["layup1", "layup2", "layup3"], help="Layup folder name")
    parser.add_argument("--window-size", type=int, default=1024, help="Number of samples per input window")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--output-path", type=str, default="model.pth", help="Path to save trained model")
    args = parser.parse_args()

    train(args)