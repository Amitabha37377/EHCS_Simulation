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
save_folder = "comparison_logsigmoid_relu"

os.makedirs(save_folder, exist_ok=True)

hls = 512

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 11
})

# =========================
# MODELS
# =========================


class NetSigmoid(nn.Module):
    def __init__(self):
        super(NetSigmoid, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(3, hls),
            nn.LogSigmoid(),
            nn.Linear(hls, hls),
            nn.LogSigmoid(),
            nn.Linear(hls, 2)
        )

    def forward(self, x):
        return self.model(x)


class NetReLU(nn.Module):
    def __init__(self):
        super(NetReLU, self).__init__()

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
# LOAD MODELS
# =========================


# Sigmoid model
checkpoint_sig = torch.load("LogSigmoid_512model.pth")
net_sig = NetSigmoid()
net_sig.load_state_dict(checkpoint_sig['model_state_dict'])
net_sig.eval()

# ReLU model
checkpoint_relu = torch.load("model_512_ReLU.pth")
net_relu = NetReLU()
net_relu.load_state_dict(checkpoint_relu['model_state_dict'])
net_relu.eval()

# Use normalization from sigmoid model (assumed same)
x_mean = torch.tensor(checkpoint_sig['x_mean'], dtype=torch.float32)
x_std = torch.tensor(checkpoint_sig['x_std'], dtype=torch.float32)
y_mean = torch.tensor(checkpoint_sig['y_mean'], dtype=torch.float32)
y_std = torch.tensor(checkpoint_sig['y_std'], dtype=torch.float32)

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

    # =========================
    # ANN ROLLOUT (BOTH MODELS)
    # =========================

    y_pred_sig, v_pred_sig = [], []
    y_pred_relu, v_pred_relu = [], []

    y_t_sig, v_t_sig = data[0, 1], data[0, 2]
    y_t_relu, v_t_relu = data[0, 1], data[0, 2]

    with torch.no_grad():

        for i in range(len(t)):

            u_t = voltage[i]

            # ---- Sigmoid ----
            inp_sig = torch.tensor(
                [[y_t_sig, v_t_sig, u_t]], dtype=torch.float32)
            inp_sig = (inp_sig - x_mean) / x_std

            out_sig = net_sig(inp_sig)
            out_sig = out_sig * y_std + y_mean

            y_next_sig = out_sig[0, 0].item()
            v_next_sig = out_sig[0, 1].item()

            y_pred_sig.append(y_next_sig)
            v_pred_sig.append(v_next_sig)

            y_t_sig, v_t_sig = y_next_sig, v_next_sig

            # ---- ReLU ----
            inp_relu = torch.tensor(
                [[y_t_relu, v_t_relu, u_t]], dtype=torch.float32)
            inp_relu = (inp_relu - x_mean) / x_std

            out_relu = net_relu(inp_relu)
            out_relu = out_relu * y_std + y_mean

            y_next_relu = out_relu[0, 0].item()
            v_next_relu = out_relu[0, 1].item()

            y_pred_relu.append(y_next_relu)
            v_pred_relu.append(v_next_relu)

            y_t_relu, v_t_relu = y_next_relu, v_next_relu

    y_pred_sig = np.array(y_pred_sig)
    v_pred_sig = np.array(v_pred_sig)

    y_pred_relu = np.array(y_pred_relu)
    v_pred_relu = np.array(v_pred_relu)

    # =========================
    # ERRORS (per timestep)
    # =========================

    mse_y_sig = (y_true - y_pred_sig)**2
    mse_v_sig = (v_true - v_pred_sig)**2

    mse_y_relu = (y_true - y_pred_relu)**2
    mse_v_relu = (v_true - v_pred_relu)**2

    # =========================
    # PLOT (3x2 GRID)
    # =========================

    fig, ax = plt.subplots(3, 2, figsize=(14, 12))

    fig.subplots_adjust(
        top=0.88,
        bottom=0.08,
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

    # -------------------------
    # Row 1
    # -------------------------

    # Input Voltage
    ax[0, 0].plot(t, voltage, linewidth=2)
    ax[0, 0].set_title("Input Voltage")
    ax[0, 0].set_xlabel("Time (s)")
    ax[0, 0].set_ylabel("Voltage (V)")
    ax[0, 0].grid(True)

    # Displacement comparison
    ax[0, 1].plot(t, y_true, label="True", linewidth=2)
    ax[0, 1].plot(t, y_pred_sig, "--", label="LogSigmoid", linewidth=2)
    ax[0, 1].plot(t, y_pred_relu, "-.", label="ReLU", linewidth=2)
    ax[0, 1].set_title("Displacement Comparison")
    ax[0, 1].set_xlabel("Time (s)")
    ax[0, 1].set_ylabel("Displacement")
    ax[0, 1].legend()
    ax[0, 1].grid(True)

    # -------------------------
    # Row 2
    # -------------------------

    # Velocity error
    ax[1, 0].plot(t, mse_v_sig, label="LogSigmoid", linewidth=2)
    ax[1, 0].plot(t, mse_v_relu, label="ReLU", linewidth=2)
    ax[1, 0].set_title("Velocity Error")
    ax[1, 0].set_xlabel("Time (s)")
    ax[1, 0].set_ylabel("Error")
    ax[1, 0].legend()
    ax[1, 0].grid(True)

    # Velocity comparison
    ax[1, 1].plot(t, v_true, label="True", linewidth=2)
    ax[1, 1].plot(t, v_pred_sig, "--", label="LogSigmoid", linewidth=2)
    ax[1, 1].plot(t, v_pred_relu, "-.", label="ReLU", linewidth=2)
    ax[1, 1].set_title("Velocity Comparison")
    ax[1, 1].set_xlabel("Time (s)")
    ax[1, 1].set_ylabel("Velocity")
    ax[1, 1].legend()
    ax[1, 1].grid(True)

    # -------------------------
    # Row 3
    # -------------------------

    # Displacement error (Sigmoid)
    ax[2, 0].plot(t, mse_y_sig, label="LogSigmoid", linewidth=2)
    ax[2, 0].set_title("Displacement Error (LogSigmoid)")
    ax[2, 0].set_xlabel("Time (s)")
    ax[2, 0].set_ylabel("Error")
    ax[2, 0].grid(True)

    # Displacement error (ReLU)
    ax[2, 1].plot(t, mse_y_relu, label="ReLU", linewidth=2)
    ax[2, 1].set_title("Displacement Error (ReLU)")
    ax[2, 1].set_xlabel("Time (s)")
    ax[2, 1].set_ylabel("Error")
    ax[2, 1].grid(True)

    # =========================
    # SAVE FIGURE
    # =========================

    out_name = os.path.basename(file).replace(".txt", ".png")
    save_path = os.path.join(save_folder, out_name)

    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved:", save_path)
