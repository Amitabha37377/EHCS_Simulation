import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.loadtxt("../preprocessing/merged_dataset.txt", skiprows=1)

x = data[:, [0, 1, 2]]  # y(t), v(t), voltage
y = data[:, [3, 4]]  # y(t+1), v(t+1)

x = torch.tensor(x, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Normalization
x_mean = x.mean(dim=0)
x_std = x.std(dim=0)
x = (x - x_mean) / x_std

y_mean = y.mean(dim=0)
y_std = y.std(dim=0)
y = (y - y_mean) / y_std

dataset = torch.utils.data.TensorDataset(x, y)
loader = torch.utils.data.DataLoader(dataset, batch_size=50, shuffle=True)

# Model

hls = 128


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, hls),
            nn.ReLU(),
            nn.Linear(hls, hls),
            nn.ReLU(),
            nn.Linear(hls, 2)
        )

    def forward(self, a):
        return self.model(a)


net = Net()

criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.000001)

# training
epochs = 150
loss_history = []

for epoch in range(epochs):
    total_loss = 0

    for batch_x, batch_y in loader:
        optimizer.zero_grad()

        outputs = net(batch_x)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    loss_history.append(avg_loss)

    if epoch % 1 == 0:
        print(f"Iteration {epoch}, Loss: {avg_loss}")

# =========================
# SAVE MODEL + NORMALIZATION
# =========================
torch.save({
    'model_state_dict': net.state_dict(),
    'x_mean': x_mean,
    'x_std': x_std,
    'y_mean': y_mean,
    'y_std': y_std
}, "model.pth")

print("Model saved successfully!")

# PLOT MSE vs EPOCH
plt.plot(loss_history)
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")
plt.title("Training Loss vs Iteration")
plt.grid()
plt.show()
