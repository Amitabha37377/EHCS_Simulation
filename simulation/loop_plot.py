import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# INPUT SIGNAL PARAMETERS
amplitudes = [1, 5, 7, 10]
frequencies = [0.1, 0.5, 1, 2]
phases = [np.pi/2, np.pi, 3*np.pi/2]

# SYSTEM PARAMETERS
m = 2.5           # Mass (kg)
k = 20000         # Spring stiffness (N/m)
Ps = 120e5        # Supply pressure (Pa)
kv = 2e-2
E = 1e8           # Bulk modulus (Pa)
K = 2.595e-7
CL = 2.595e-9     # Leakage coefficient

# Cylinder geometry
dc = 0.04
dr = 0.002

Aa = np.pi * dc**2 / 4
Ab = np.pi * (dc**2 - dr**2) / 4
A = (Aa + Ab) / 2

V0a = Aa * 0.1
V0b = Ab * 0.1
Vt = V0a + V0b

# FRICTION MODEL


def friction(v):
    Fc = 51.1
    alpha = 2000
    return Fc * np.sign(v) + alpha * v


for Vm in amplitudes:
    for freq in frequencies:
        for phase in phases:

            # INITIAL CONDITIONS
            y = 0
            yd = 0
            Pa = Ps / 2
            Pb = Ps / 2

            # VOLTAGE INPUT

            def square_voltage(t):
                return Vm * np.sign(np.sin(2 * np.pi * freq * t + phase))

            def sine_voltage(t, Vm=Vm, f=freq, phi=phase):
                return Vm * np.sin(2 * np.pi * f * t + phi)

            # SIMULATION SETUP
            dt = 1e-4
            t_end = 20
            t_span = np.arange(0, t_end, dt)

            # Storage arrays
            y_arr = [y]
            Pa_arr = [Pa]
            Pb_arr = [Pb]
            V_arr = []

            # TIME INTEGRATION (Euler Method)
            for t in t_span:

                V = square_voltage(t)
                V_arr.append(V)

                Pl = Pa - Pb

                QL = V * K * np.sqrt(max((Ps - np.sign(V) * Pl), 0) / 2)
                dPl = 4 * E * (QL - A * yd - CL * Pl) / Vt
                ydd = (Pl * A - friction(yd) - k * y) / m

                # Euler integration
                yd += ydd * dt
                y += yd * dt
                Pl += dPl * dt

                # Update pressures
                Pa = Ps/2 + Pl/2
                Pb = Ps/2 - Pl/2

                # Store states
                y_arr.append(y)
                Pa_arr.append(Pa)
                Pb_arr.append(Pb)

            # POST-PROCESSING
            y_arr = np.clip(np.array(y_arr), -0.1, 0.1)
            Pa_arr = np.array(Pa_arr)
            Pb_arr = np.array(Pb_arr)
            V_arr = np.array(V_arr)

            # Velocity from displacement (finite difference)
            yd_arr = np.zeros_like(y_arr)
            yd_arr[1:] = np.diff(y_arr) / dt

            # # Smoothing
            # fs = 1/dt
            # cutoff = 10  # Hz
            # b, a = butter(4, cutoff/(fs/2), btype='low')
            #
            # yd_arr = filtfilt(b, a, yd_arr)
            # y_arr = filtfilt(b, a, y_arr)

            # Plotting
            t_plot = np.arange(0, t_end + dt, dt)
            phase_deg = np.degrees(phase)

            fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex='col')

            fig.suptitle(
                f"Electrohydraulic Actuator Response: Square Voltage\n"
                f"Amplitude = {Vm} V | Frequency = {
                    freq} Hz | Phase = {phase_deg:.1f}°",
                fontsize=14,
                fontweight='bold',
                y=0.95
            )

            # Voltage
            axs[0, 0].plot(t_span, V_arr)
            axs[0, 0].set_title("Input Voltage")
            axs[0, 0].set_ylabel("Voltage (V)")
            axs[0, 0].grid(True)

            # Pressure
            axs[1, 0].plot(t_plot, Pa_arr, label="Pa")
            axs[1, 0].plot(t_plot, Pb_arr, label="Pb")
            axs[1, 0].set_title("Chamber Pressures")
            axs[1, 0].set_ylabel("Pressure (Pa)")
            axs[1, 0].set_xlabel("Time (s)")
            axs[1, 0].legend()
            axs[1, 0].grid(True)

            # Displacement
            axs[0, 1].plot(t_plot, y_arr)
            axs[0, 1].set_title("Displacement")
            axs[0, 1].set_ylabel("Displacement (m)")
            axs[0, 1].grid(True)

            # Velocity
            axs[1, 1].plot(t_plot, yd_arr)
            axs[1, 1].set_title("Velocity")
            axs[1, 1].set_ylabel("Velocity (m/s)")
            axs[1, 1].set_xlabel("Time (s)")
            axs[1, 1].grid(True)

            fig.subplots_adjust(
                top=0.82,
                bottom=0.10,
                left=0.08,
                right=0.97,
                wspace=0.25,
                hspace=0.35
            )

            # plt.show()
            phase_deg = int(np.degrees(phase))

            filename = f"Square_V{Vm}_f{freq}_ph{phase_deg}.png"

            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"Plot saved as: {filename}")
