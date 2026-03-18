import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# =========================
# SETTINGS
# =========================

data_folder = "../preprocessing/data_downsampled/"
save_folder = "comparison_plots"

os.makedirs(save_folder, exist_ok=True)

hls = 128


# =========================
# MODEL
# =========================

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
# LOAD MODEL
# =========================

checkpoint = torch.load("model.pth")

net = Net()
net.load_state_dict(checkpoint['model_state_dict'])
net.eval()

x_mean = torch.tensor(checkpoint['x_mean'], dtype=torch.float32)
x_std = torch.tensor(checkpoint['x_std'], dtype=torch.float32)
y_mean = torch.tensor(checkpoint['y_mean'], dtype=torch.float32)
y_std = torch.tensor(checkpoint['y_std'], dtype=torch.float32)


# =========================
# PROCESS EACH FILE
# =========================

files = glob.glob(os.path.join(data_folder, "*.txt"))

for file in files:

    data = np.loadtxt(file, skiprows=5)

    t = data[:, 0]
    y_true = data[:, 3]     # y(t+1)
    v_true = data[:, 4]     # v(t+1)
    voltage = data[:, 5]

    # initial state
    y_t = data[0, 1]
    v_t = data[0, 2]

    y_pred = []
    v_pred = []

    # =========================
    # ANN ROLLOUT
    # =========================

    with torch.no_grad():

        for i in range(len(t)):

            u_t = voltage[i]

            inp = torch.tensor([[y_t, v_t, u_t]], dtype=torch.float32)

            inp = (inp - x_mean) / x_std

            out = net(inp)

            out = out * y_std + y_mean

            y_next = out[0, 0].item()
            v_next = out[0, 1].item()

            y_pred.append(y_next)
            v_pred.append(v_next)

            y_t = y_next
            v_t = v_next

    y_pred = np.array(y_pred)
    v_pred = np.array(v_pred)

    # =========================
    # MSE PER TIME STEP
    # =========================

    mse_y = (y_true - y_pred)**2
    mse_v = (v_true - v_pred)**2

    # =========================
    # PLOT
    # =========================

    plt.figure(figsize=(10, 8))

    # Voltage
    plt.subplot(4, 1, 1)
    plt.plot(t, voltage)
    plt.title("Voltage")
    plt.ylabel("V")
    plt.grid()

    # Displacement
    plt.subplot(4, 1, 2)
    plt.plot(t, y_true, label="True")
    plt.plot(t, y_pred, "--", label="ANN")
    plt.title("Displacement Comparison")
    plt.ylabel("y")
    plt.legend()
    plt.grid()

    # Velocity
    plt.subplot(4, 1, 3)
    plt.plot(t, v_true, label="True")
    plt.plot(t, v_pred, "--", label="ANN")
    plt.title("Velocity Comparison")
    plt.ylabel("v")
    plt.legend()
    plt.grid()

    # MSE
    plt.subplot(4, 1, 4)
    plt.plot(t, mse_y, label="Displacement MSE")
    plt.plot(t, mse_v, label="Velocity MSE")
    plt.title("MSE vs Time")
    plt.xlabel("Time")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid()

    plt.tight_layout()

    # =========================
    # SAVE FIGURE
    # =========================

    filename = os.path.basename(file).replace(".txt", ".png")
    save_path = os.path.join(save_folder, filename)

    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved:", save_path)
