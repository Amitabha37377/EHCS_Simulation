import numpy as np
import os
import glob

# ===== SETTINGS =====
input_folder = "../data_2cycles"
output_folder = "./data_downsampled"
factor = 100

os.makedirs(output_folder, exist_ok=True)

# ===== PROCESS EACH FILE =====
files = glob.glob(os.path.join(input_folder, "*.txt"))

for filepath in files:
    filename = os.path.basename(filepath)
    output_path = os.path.join(output_folder, filename)

    # ---- READ HEADER (first 3 lines) ----
    with open(filepath, 'r') as f:
        header_lines = [next(f) for _ in range(3)]

    # ---- LOAD NUMERIC DATA ----
    data = np.loadtxt(filepath, skiprows=6)

    # Columns
    time = data[:, 0]
    y = data[:, 1]
    v = data[:, 2]
    voltage = data[:, 5]

    # ---- DOWNSAMPLE ----
    time_ds = time[::factor]
    y_ds = y[::factor]
    v_ds = v[::factor]
    voltage_ds = voltage[::factor]

    # ---- REBUILD (t → t+1) PAIRS ----
    time_final = time_ds[:-1]
    y_t = y_ds[:-1]
    v_t = v_ds[:-1]
    y_t1 = y_ds[1:]
    v_t1 = v_ds[1:]
    voltage_final = voltage_ds[:-1]

    # ---- WRITE FILE ----
    with open(output_path, 'w') as f:
        # Write header exactly
        for line in header_lines:
            f.write(line)

        # Blank line like original
        f.write("\n")

        # Column headers (aligned)
        f.write(f"{'time':>12} {'y(t)':>15} {'v(t)':>15} {
                'y(t+1)':>15} {'v(t+1)':>15} {'Voltage':>15}\n")

        # Write data with proper alignment
        for i in range(len(time_final)):
            f.write(
                f"{time_final[i]:12.6f} "
                f"{y_t[i]:15.9f} "
                f"{v_t[i]:15.9f} "
                f"{y_t1[i]:15.9f} "
                f"{v_t1[i]:15.9f} "
                f"{voltage_final[i]:15.9f}\n"
            )

    print(f"Saved: {output_path}")
