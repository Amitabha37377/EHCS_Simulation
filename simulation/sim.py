
import numpy as np
import matplotlib.pyplot as plt

amplitudes = [1, 5, 7, 10]
phi = [np.pi/2, np.pi, np.pi*1.5]
frequencies = [0.1, 0.5, 1, 2]

# Parametric values
m = 2.5  # kg
Aa = np.pi * (0.04 * 0.04)/4
Ab = np.pi * (0.04**2 - 0.002**2)/4
k = 20000
Ps = 120e5
kv = 2e-2
E = 1e8
K = 2.595e-7
# CL = 0
CL = 2.595e-9
V_0a = Aa * 0.1
V_0b = Ab * 0.1

A = (Aa + Ab)/2
Vt = V_0a + V_0b

# initial condition
y = 0
yd = 0
Pa = Ps/2
Pb = Ps/2

# Frinction


def f(v):
    Fc = 51.1
    alpha = 2000
    return Fc * np.sign(v) + alpha * v


Vm = 6
freq = 0.5
phase = 0


def sine_voltage(t, Vm=Vm, f=freq, phi=phase):
    return Vm * np.sin(2 * np.pi * f * t + phi)


def square_voltage(t, Vm=Vm, f=freq, phi=phase):
    return Vm * np.sign(np.sin(2 * np.pi * f * t + phi))


# simulation steps
dt = (2/freq) / 200000
t_span = np.arange(0, 2/freq, dt)

y_arr = [y]
yd_arr = [yd]
Pa_arr = [Pa]
Pb_arr = [Pb]
V_arr = []

# print(f"{'y(t)':>10} {'v(t)':>10} {'y(t+1)':>10} {'v(t+1)':>10} {'V':>10}")

for t in t_span:
    # V = square_voltage(t)
    V = sine_voltage(t)
    z = kv * V
    Pl = Pa - Pb
    V_arr.append(V)

    X_1 = [y, yd]

    QL = V * K * np.sqrt(max((Ps - (np.sign(z) * Pl)), 0)/2)
    dPl = 4 * E * (QL - A * yd - CL * Pl) / Vt
    ydd = (Pl*A - f(yd) - k*y)/m

    yd = yd + ydd * dt
    y = y + yd * dt
    Pl = Pl + dPl * dt

    P0 = Ps / 2
    Pa = P0 + Pl/2
    Pb = P0 - Pl/2

    X_2 = [y, yd]

    y_arr.append(y)
    yd_arr.append(yd)
    Pa_arr.append(Pa)
    Pb_arr.append(Pb)

# Plotting
# Convert lists to numpy arrays
y_arr = np.array(y_arr)
yd_arr = np.array(yd_arr)
Pa_arr = np.array(Pa_arr)
Pb_arr = np.array(Pb_arr)
V_arr = np.array(V_arr)

y_arr = np.clip(y_arr, -0.1, 0.1)
yd_arr = np.zeros_like(y_arr)   # initialize with zeros
yd_arr[1:] = (y_arr[1:] - y_arr[:-1]) / dt

t_plot = np.arange(0, 2/freq + dt, dt)

phase_deg = np.degrees(phase)

fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex='col')

fig.suptitle(
    f"Electrohydraulic Actuator Response: Sinosoidal Voltage\n"
    f"Amplitude = {Vm} V | Frequency = {freq} Hz | Phase = {phase_deg:.1f}°",
    fontsize=14,
    fontweight='bold',
    y=0.96
)

# ---------------- LEFT COLUMN ----------------
# Voltage (Top-Left)
axs[0, 0].plot(t_plot[:-1], V_arr)
axs[0, 0].set_title("Input Voltage")
axs[0, 0].set_ylabel("Voltage (V)")
axs[0, 0].grid(True)

# Pressure (Bottom-Left)
axs[1, 0].plot(t_plot, Pa_arr, label="Pa")
axs[1, 0].plot(t_plot, Pb_arr, label="Pb")
axs[1, 0].set_title("Chamber Pressures")
axs[1, 0].set_ylabel("Pressure (Pa)")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].legend()
axs[1, 0].grid(True)

# ---------------- RIGHT COLUMN ----------------
# Displacement (Top-Right)
axs[0, 1].plot(t_plot, y_arr)
axs[0, 1].set_title("Displacement")
axs[0, 1].set_ylabel("Displacement (m)")
axs[0, 1].grid(True)

# Velocity (Bottom-Right)
axs[1, 1].plot(t_plot, yd_arr)
axs[1, 1].set_title("Velocity")
axs[1, 1].set_ylabel("Velocity (m/s)")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].grid(True)

# Adjust spacing
fig.subplots_adjust(
    top=0.80,
    bottom=0.10,
    left=0.08,
    right=0.97,
    wspace=0.25,
    hspace=0.35
)

plt.show()
