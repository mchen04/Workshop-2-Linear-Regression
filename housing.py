import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
import torch
import torch.nn as nn
import torch.optim as optim

# Load California housing dataset
california = fetch_california_housing()

# Randomly select 1000 data points
indices = np.random.choice(range(len(california.target)), 1000, replace=False)
X_california = california.data[indices, 0].reshape(-1, 1)  # Median income
y_california = california.target[indices]

# Plot the data points before training
plt.scatter(X_california, y_california, color='blue', label='Data points')
plt.xlabel('Median Income')
plt.ylabel('Housing Price')
plt.title('California Housing Prices (Before Training)')
plt.legend()
plt.show()

# Define the number of epochs
epochs = 100

# Model, Loss Function, and Optimizer
model = nn.Linear(1, 1)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

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

# Plotting after training
with torch.no_grad():
    predicted = model(torch.from_numpy(X_california).float()).numpy()
plt.scatter(X_california, y_california, color='blue', label='Data points')
plt.plot(X_california, predicted, color='red', label='Regression Line')
plt.xlabel('Median Income')
plt.ylabel('Housing Price')
plt.title('California Housing Prices (After Training)')
plt.legend()
plt.show()