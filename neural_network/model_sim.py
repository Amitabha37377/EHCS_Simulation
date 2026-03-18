import torch
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. DEFINE MODEL (same as training)
# =========================
import torch.nn as nn

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

    def forward(self, x):
        return self.model(x)


# =========================
# 2. LOAD MODEL
# =========================
checkpoint = torch.load("model.pth")

net = Net()
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

x_mean = checkpoint['x_mean']
x_std = checkpoint['x_std']
y_mean = checkpoint['y_mean']
y_std = checkpoint['y_std']

# =========================
# 3. GENERATE VOLTAGE SIGNAL
# =========================
A = 10
f = 0.1
phi = np.pi / 2   # 90 degrees

T = 1 / f
total_time = 2 * T   # 2 cycles

dt = 0.01
t = np.arange(0, total_time, dt)

voltage = A * np.sin(2 * np.pi * f * t + phi)

# =========================
# 4. INITIAL CONDITIONS
# =========================
y_t = 0.0
v_t = 0.0

y_list = []
v_list = []

# =========================
# 5. ROLLOUT (IMPORTANT PART)
# =========================
with torch.no_grad():
    for i in range(len(t)):

        u_t = voltage[i]

        inp = torch.tensor([[y_t, v_t, u_t]], dtype=torch.float32)

        # Normalize
        inp = (inp - x_mean) / x_std

        # Predict
        out = net(inp)

        # Denormalize
        out = out * y_std + y_mean

        y_next = out[0, 0].item()
        v_next = out[0, 1].item()

        # Store
        y_list.append(y_next)
        v_list.append(v_next)

        # FEEDBACK (critical)
        y_t = y_next
        v_t = v_next

# =========================
# 6. PLOT RESULTS (ONE WINDOW, 3 SUBPLOTS)
# =========================

plt.figure(figsize=(10, 8))

# Voltage
plt.subplot(3, 1, 1)
plt.plot(t, voltage)
plt.title("Input Voltage")
plt.ylabel("Voltage")
plt.grid()

# Displacement
plt.subplot(3, 1, 2)
plt.plot(t, y_list)
plt.title("Displacement y(t)")
plt.ylabel("Displacement")
plt.grid()

# Velocity
plt.subplot(3, 1, 3)
plt.plot(t, v_list)
plt.title("Velocity v(t)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity")
plt.grid()

plt.tight_layout()
plt.show()
