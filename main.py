# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Data Preparation
# Generate random data
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Plot
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Synthetic Linear Data')
plt.show()

# Model Creation
# Define a linear model
model = nn.Linear(1, 1)  # 1 input feature, 1 output feature

# Define loss function and optimizer
criterion = torch.nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Training the Model
epochs = 100
for epoch in range(epochs):
    # Convert arrays to tensors
    inputs = torch.from_numpy(X).float()
    targets = torch.from_numpy(y).float()

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Generate predictions from the model
with torch.no_grad():
    predicted = model(torch.from_numpy(X).float()).data.numpy()

# Plot the linear regression line
plt.scatter(X, y)
plt.plot(X, predicted, color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Result')
plt.legend()
plt.show()