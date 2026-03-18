import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import butter, filtfilt
import sys

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

                V = sine_voltage(t)
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

            # Printing data
            phase_deg = phase * 180 / np.pi
            filename = f"Sine_V{Vm}_f{freq}_ph{phase_deg}.txt"
            file = open(filename, "w")
            sys.stdout = file

            print("Voltage Amplitude: ", Vm, " Volts")
            print("Voltage Frequency: ", freq, " Hz")
            print("Phase Difference: ", phase/np.pi * 180, " Degrees")
            print("\n")

            print(f"{'time':>10} {'y(t)':>15} {'v(t)':>15} {
                  'y(t+1)':>15} {'v(t+1)':>15} {'Voltage':>12}")

            N = len(t_span)

            for i in range(N):
                t_current = t_span[i]
                V_current = sine_voltage(t_current)

                print(f"{t_current:10.6f} "
                      f"{y_arr[i]:15.9f} "
                      f"{yd_arr[i]:15.9f} "
                      f"{y_arr[i+1]:15.9f} "
                      f"{yd_arr[i+1]:15.9f} "
                      f"{V_current:12.6f}")

            file.close()
