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

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 11
})

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
# PROCESS FILES
# =========================

files = glob.glob(os.path.join(data_folder, "*.txt"))

for file in files:

    print("Processing:", file)

    # -------------------------
    # READ HEADER INFO
    # -------------------------

    with open(file, "r") as f:
        header = [next(f) for _ in range(3)]

    amp = header[0].split(":")[1].strip().split()[0]
    freq = header[1].split(":")[1].strip().split()[0]
    phase = header[2].split(":")[1].strip().split()[0]

    # -------------------------
    # DETECT WAVE TYPE
    # -------------------------

    filename = os.path.basename(file)

    if "Sine" in filename or "sine" in filename:
        wave_type = "Sine"
    elif "Square" in filename or "square" in filename:
        wave_type = "Square"
    else:
        wave_type = "Unknown"

    # -------------------------
    # LOAD DATA
    # -------------------------

    data = np.loadtxt(file, skiprows=5)

    t = data[:, 0]
    y_true = data[:, 3]
    v_true = data[:, 4]
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
    # MSE
    # =========================

    mse_y = (y_true - y_pred)**2
    mse_v = (v_true - v_pred)**2

    # =========================
    # PLOT (2x2 GRID)
    # =========================

    fig, ax = plt.subplots(2, 2, figsize=(14, 8))

    fig.subplots_adjust(
        top=0.82,
        bottom=0.10,
        left=0.08,
        right=0.97,
        wspace=0.25,
        hspace=0.35
    )

    fig.suptitle(
        f"Physics based simulation vs Neural network output Comparison\n"
        f"{wave_type} Voltage | Amplitude = {
            amp} V | Frequency = {freq} Hz | Phase = {phase}°",
        fontsize=18,
        fontweight="bold"
    )

    # Voltage
    ax[0, 0].plot(t, voltage, linewidth=2)
    ax[0, 0].set_title("Input Voltage")
    ax[0, 0].set_xlabel("Time (s)")
    ax[0, 0].set_ylabel("Voltage (V)")
    ax[0, 0].grid(True)

    # Displacement comparison
    ax[0, 1].plot(t, y_true, label="True", linewidth=2)
    ax[0, 1].plot(t, y_pred, "--", label="ANN", linewidth=2)

    ax[0, 1].set_title("Displacement Comparison")
    ax[0, 1].set_xlabel("Time (s)")
    ax[0, 1].set_ylabel("Displacement")
    ax[0, 1].legend()
    ax[0, 1].grid(True)

    # MSE Errors
    ax[1, 0].plot(t, mse_y, label="Displacement MSE", linewidth=2)
    ax[1, 0].plot(t, mse_v, label="Velocity MSE", linewidth=2)

    ax[1, 0].set_title("MSE Errors")
    ax[1, 0].set_xlabel("Time (s)")
    ax[1, 0].set_ylabel("MSE")
    ax[1, 0].legend()
    ax[1, 0].grid(True)

    # Velocity comparison
    ax[1, 1].plot(t, v_true, label="True", linewidth=2)
    ax[1, 1].plot(t, v_pred, "--", label="ANN", linewidth=2)

    ax[1, 1].set_title("Velocity Comparison")
    ax[1, 1].set_xlabel("Time (s)")
    ax[1, 1].set_ylabel("Velocity")
    ax[1, 1].legend()
    ax[1, 1].grid(True)

    # plt.tight_layout(rect=[0, 0, 1, 0.92])

    # =========================
    # SAVE FIGURE
    # =========================

    out_name = os.path.basename(file).replace(".txt", ".png")
    save_path = os.path.join(save_folder, out_name)

    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved:", save_path)

# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# import glob
#
# # =========================
# # SETTINGS
# # =========================
#
# data_folder = "../preprocessing/data_downsampled/"
# save_folder = "comparison_plots"
#
# os.makedirs(save_folder, exist_ok=True)
#
# hls = 128
#
# plt.rcParams.update({
#     "font.size": 12,
#     "axes.titlesize": 13,
#     "axes.labelsize": 11
# })
#
# # =========================
# # MODEL
# # =========================
#
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#
#         self.model = nn.Sequential(
#             nn.Linear(3, hls),
#             nn.ReLU(),
#             nn.Linear(hls, hls),
#             nn.ReLU(),
#             nn.Linear(hls, 2)
#         )
#
#     def forward(self, x):
#         return self.model(x)
#
#
# # =========================
# # LOAD MODEL
# # =========================
#
# checkpoint = torch.load("model.pth")
#
# net = Net()
# net.load_state_dict(checkpoint['model_state_dict'])
# net.eval()
#
# # Normalization parameters
# x_mean = torch.tensor(checkpoint['x_mean'], dtype=torch.float32)
# x_std = torch.tensor(checkpoint['x_std'], dtype=torch.float32)
# y_mean = torch.tensor(checkpoint['y_mean'], dtype=torch.float32)
# y_std = torch.tensor(checkpoint['y_std'], dtype=torch.float32)
#
# # =========================
# # PROCESS FILES
# # =========================
#
# files = glob.glob(os.path.join(data_folder, "*.txt"))
#
# for file in files:
#
#     print("Processing:", file)
#
#     # -------------------------
#     # READ SIGNAL PARAMETERS
#     # -------------------------
#
#     with open(file, "r") as f:
#         header = [next(f) for _ in range(3)]
#
#     amp = header[0].split(":")[1].strip().split()[0]
#     freq = header[1].split(":")[1].strip().split()[0]
#     phase = header[2].split(":")[1].strip().split()[0]
#
#     # -------------------------
#     # LOAD DATA
#     # -------------------------
#
#     data = np.loadtxt(file, skiprows=5)
#
#     t = data[:, 0]
#     y_true = data[:, 3]
#     v_true = data[:, 4]
#     voltage = data[:, 5]
#
#     # initial state
#     y_t = data[0, 1]
#     v_t = data[0, 2]
#
#     y_pred = []
#     v_pred = []
#
#     # =========================
#     # ANN ROLLOUT
#     # =========================
#
#     with torch.no_grad():
#
#         for i in range(len(t)):
#
#             u_t = voltage[i]
#
#             inp = torch.tensor([[y_t, v_t, u_t]], dtype=torch.float32)
#
#             inp = (inp - x_mean) / x_std
#
#             out = net(inp)
#
#             out = out * y_std + y_mean
#
#             y_next = out[0, 0].item()
#             v_next = out[0, 1].item()
#
#             y_pred.append(y_next)
#             v_pred.append(v_next)
#
#             y_t = y_next
#             v_t = v_next
#
#     y_pred = np.array(y_pred)
#     v_pred = np.array(v_pred)
#
#     # =========================
#     # MSE
#     # =========================
#
#     mse_y = (y_true - y_pred)**2
#     mse_v = (v_true - v_pred)**2
#
#     # =========================
#     # PLOT (2x2 GRID)
#     # =========================
#
#     fig, ax = plt.subplots(2, 2, figsize=(14, 8))
#
#     fig.suptitle(
#         f"Electrohydraulic Actuator Response\n"
#         f"Voltage = {amp} V | Frequency = {freq} Hz | Phase = {phase}°",
#         fontsize=18,
#         fontweight="bold"
#     )
#
#     # -------------------------
#     # Voltage (Top Left)
#     # -------------------------
#
#     ax[0, 0].plot(t, voltage, linewidth=2)
#     ax[0, 0].set_title("Input Voltage")
#     ax[0, 0].set_xlabel("Time (s)")
#     ax[0, 0].set_ylabel("Voltage (V)")
#     ax[0, 0].grid(True)
#
#     # -------------------------
#     # Displacement Comparison
#     # -------------------------
#
#     ax[0, 1].plot(t, y_true, label="True", linewidth=2)
#     ax[0, 1].plot(t, y_pred, "--", label="ANN", linewidth=2)
#
#     ax[0, 1].set_title("Displacement Comparison")
#     ax[0, 1].set_xlabel("Time (s)")
#     ax[0, 1].set_ylabel("Displacement (m)")
#     ax[0, 1].legend()
#     ax[0, 1].grid(True)
#
#     # -------------------------
#     # MSE Errors (Bottom Left)
#     # -------------------------
#
#     ax[1, 0].plot(t, mse_y, label="Displacement MSE", linewidth=2)
#     ax[1, 0].plot(t, mse_v, label="Velocity MSE", linewidth=2)
#
#     ax[1, 0].set_title("MSE Errors")
#     ax[1, 0].set_xlabel("Time (s)")
#     ax[1, 0].set_ylabel("MSE")
#     ax[1, 0].legend()
#     ax[1, 0].grid(True)
#
#     # -------------------------
#     # Velocity Comparison
#     # -------------------------
#
#     ax[1, 1].plot(t, v_true, label="True", linewidth=2)
#     ax[1, 1].plot(t, v_pred, "--", label="ANN", linewidth=2)
#
#     ax[1, 1].set_title("Velocity Comparison")
#     ax[1, 1].set_xlabel("Time (s)")
#     ax[1, 1].set_ylabel("Velocity (m/s)")
#     ax[1, 1].legend()
#     ax[1, 1].grid(True)
#
#     plt.tight_layout(rect=[0, 0, 1, 0.92])
#
#     # =========================
#     # SAVE FIGURE
#     # =========================
#
#     filename = os.path.basename(file).replace(".txt", ".png")
#     save_path = os.path.join(save_folder, filename)
#
#     plt.savefig(save_path, dpi=300)
#     plt.close()
#
#     print("Saved:", save_path)
