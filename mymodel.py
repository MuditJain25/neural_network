import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# --- 1. The Model Architecture ---
class MultiprocessingMLP(nn.Module):
    def __init__(self):
        super(MultiprocessingMLP, self).__init__()
        # Input Layer (A0) to Hidden Layer (A1)
        # 784 nodes for the 28x28 normalized pixels mapped to 256 neurons
        self.fc1 = nn.Linear(784, 256) 
        
        # Hidden Layer (A1) to Output Layer (A2)
        # 256 neurons mapped to 10 output digit predictions
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # Flatten the 28x28 image into a 1D array of 784 pixels
        x = torch.flatten(x, 1)
        
        # Pass through Hidden Layer and apply ReLU activation
        x = self.fc1(x)
        x = F.relu(x)
        
        # Pass to Output Layer
        x = self.fc2(x)
        return x

# --- 2. The Training Routine ---
def main():
    # Setup Device (Use GPU if you have one, otherwise fallback to CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Prepare Data
    # PyTorch will create a folder called 'data' and handle the idx files there
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # Using batch_size=64 for standard training speed. 
    # Change to batch_size=1 if you want strict adherence to your document's SGD definition.
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize Model, Loss, and Optimizer
    model = MultiprocessingMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training Loop
    epochs = 5
    print("Starting training...")
    
    for epoch in range(epochs):
        model.train() 
        running_loss = 0.0
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            # Move data to the correct device (CPU or GPU)
            data, labels = data.to(device), labels.to(device)
            
            # Step A: Zero out previous gradients
            optimizer.zero_grad()
            
            # Step B: Forward pass (Predict)
            predictions = model(data)
            
            # Step C: Calculate the loss
            loss = criterion(predictions, labels)
            
            # Step D: Backward pass (Calculate gradients)
            loss.backward()
            
            # Step E: Update the weights
            optimizer.step()
            
            running_loss += loss.item()
            
            # Print progress every 100 batches
            if batch_idx % 100 == 99:
                print(f'Epoch: {epoch + 1}/{epochs} | Batch: {batch_idx + 1} | Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

    print("Training Complete!")

if __name__ == '__main__':
    main()