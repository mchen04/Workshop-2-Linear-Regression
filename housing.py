from sklearn.datasets import fetch_california_housing
import torch
import torch.nn as nn
import torch.optim as optim

# Load California housing dataset
california = fetch_california_housing()
print(california.DESCR)

# Selecting a feature and the target
# For simplicity, let's use MedInc (median income) as our feature
X_california = california.data[:, 0].reshape(-1, 1)  # Median income
y_california = california.target

# Define the number of epochs
epochs = 100

# Model, Loss Function, and Optimizer
model = nn.Linear(1, 1)  # Model with 1 input and 1 output feature
criterion = torch.nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Training loop
for epoch in range(epochs):
    # Convert arrays to tensors
    inputs = torch.from_numpy(X_california).float()
    targets = torch.from_numpy(y_california).float()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 10 epochs
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
