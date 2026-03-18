import numpy as np
import os
import glob

# ===== SETTINGS =====
input_folder = "./data_downsampled"
output_file = "./merged_dataset.txt"

all_data = []

# ===== READ FILES =====
files = glob.glob(os.path.join(input_folder, "*.txt"))

for filepath in files:
    print(f"Processing: {filepath}")

    # Load numeric data (skip header)
    data = np.loadtxt(filepath, skiprows=5)

    # Columns:
    # time | y(t) | v(t) | y(t+1) | v(t+1) | voltage
    y_t = data[:, 1]
    v_t = data[:, 2]
    y_t1 = data[:, 3]
    v_t1 = data[:, 4]
    voltage = data[:, 5]

    # Stack: [y(t), v(t), voltage, y(t+1), v(t+1)]
    combined = np.column_stack((y_t, v_t, voltage, y_t1, v_t1))

    all_data.append(combined)

# ===== MERGE ALL FILES =====
final_data = np.vstack(all_data)

# ===== SAVE =====
with open(output_file, "w") as f:
    # Header
    f.write(f"{'y(t)':>15} {'v(t)':>15} {'voltage':>15} {
            'y(t+1)':>15} {'v(t+1)':>15}\n")

    # Data
    for row in final_data:
        f.write(
            f"{row[0]:15.9f} {row[1]:15.9f} {row[2]:15.9f} "
            f"{row[3]:15.9f} {row[4]:15.9f}\n"
        )

print(f"\nFinal dataset saved: {output_file}")
print(f"Total samples: {final_data.shape[0]}")
