import numpy as np
import matplotlib.pyplot as plt
import os
import re

# ---- SETTINGS ----
data_folder = "../data_2cycles"
output_folder = "./plots_downsampled"
factor = 100

os.makedirs(output_folder, exist_ok=True)

# ---- LOOP THROUGH FILES ----
for file in os.listdir(data_folder):

    if not file.endswith(".txt"):
        continue

    filepath = os.path.join(data_folder, file)

    # ---- LOAD DATA ----
    data = np.loadtxt(filepath, skiprows=6)

    time = data[:, 0]
    y_t = data[:, 1]
    v_t = data[:, 2]
    voltage = data[:, 5]

    # ---- DOWNSAMPLE ----
    time_ds = time[::factor]
    y_t_ds = y_t[::factor]
    v_t_ds = v_t[::factor]
    voltage_ds = voltage[::factor]

    # ---- EXTRACT INFO FROM FILENAME ----
    pattern = r"(Sine|Square)_V([\d\.]+)_f([\d\.]+)_ph([\d\.]+)\.txt"
    match = re.search(pattern, file)

    if match:
        signal_type = match.group(1)
        Vm = float(match.group(2))
        freq = float(match.group(3))
        phase = float(match.group(4))
    else:
        signal_type = "Unknown"
        Vm, freq, phase = 0, 0, 0

    # ---- PLOTTING ----
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex='col')

    fig.suptitle(
        f"Electrohydraulic Actuator Response: {signal_type} Voltage\n"
        f"Amplitude = {Vm} V | Frequency = {freq} Hz | Phase = {phase}°",
        fontsize=14,
        fontweight='bold',
        y=0.95
    )

    # Voltage
    axs[0, 0].plot(time_ds, voltage_ds)
    axs[0, 0].set_title("Input Voltage")
    axs[0, 0].set_ylabel("Voltage (V)")
    axs[0, 0].grid(True)

    # Combined y & v
    axs[1, 0].plot(time_ds, y_t_ds, label="y(t)")
    axs[1, 0].plot(time_ds, v_t_ds, label="v(t)")
    axs[1, 0].set_title("Displacement & Velocity")
    axs[1, 0].set_xlabel("Time (s)")
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Displacement
    axs[0, 1].plot(time_ds, y_t_ds)
    axs[0, 1].set_title("Displacement")
    axs[0, 1].set_ylabel("y (m)")
    axs[0, 1].grid(True)

    # Velocity
    axs[1, 1].plot(time_ds, v_t_ds)
    axs[1, 1].set_title("Velocity")
    axs[1, 1].set_xlabel("Time (s)")
    axs[1, 1].set_ylabel("v (m/s)")
    axs[1, 1].grid(True)

    # Layout
    fig.subplots_adjust(
        top=0.82,
        bottom=0.10,
        left=0.08,
        right=0.97,
        wspace=0.25,
        hspace=0.35
    )

    # ---- SAVE PLOT ----
    output_name = file.replace(".txt", ".png")
    output_path = os.path.join(output_folder, output_name)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {output_name}")
