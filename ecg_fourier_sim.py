
import sys
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# --- SIMULATION MATH & LOGIC ---
def generate_ecg(bpm, noise_amp, fs, duration):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    f0 = bpm / 60.0
    ecg = np.zeros_like(t)
    
    # Synthesize a simplified periodic ECG (QRS complex + T wave)
    period_samples = int(fs / f0)
    for i in range(0, len(t), period_samples):
        # QRS Peak
        ecg += 1.2 * np.exp(-((np.arange(len(t)) - i)**2) / (2 * (0.01 * fs)**2))
        # T Wave (slower, wider)
        t_wave_center = i + int(0.35 * fs)
        if t_wave_center < len(t):
            ecg += 0.3 * np.exp(-((np.arange(len(t)) - t_wave_center)**2) / (2 * (0.04 * fs)**2))
            
    # Noise: 50 Hz electrical hum + random high-frequency baseline wander
    hum_noise = noise_amp * 0.8 * np.sin(2 * np.pi * 50 * t)
    random_noise = noise_amp * 0.3 * np.random.randn(len(t))
    
    noisy_ecg = ecg + hum_noise + random_noise
    return t, ecg, noisy_ecg

def compute_fft(y, fs):
    N = len(y)
    Y = np.fft.fft(y)
    f = np.fft.fftfreq(N, 1/fs)
    # Return only positive frequencies
    pos_mask = f >= 0
    return f[pos_mask], (2.0/N) * np.abs(Y[pos_mask])

def design_lowpass(fc):
    # 2nd Order Butterworth Low-Pass Filter
    b, a = signal.butter(2, fc, 'low', analog=True)
    return b, a

def apply_digital_filter(y, fc, fs):
    # Apply the equivalent digital filter for time-domain plotting
    b, a = signal.butter(2, fc, 'low', fs=fs)
    return signal.filtfilt(b, a, y)

# --- GUI APPLICATION ---
def update_simulation(*args):
    bpm = slider_bpm.get()
    noise_amp = slider_noise.get()
    fc = slider_fc.get()
    
    lbl_val_bpm.config(text=f"{bpm:.0f} BPM")
    lbl_val_noise.config(text=f"{noise_amp:.2f}")
    lbl_val_fc.config(text=f"{fc:.1f} Hz")
    
    fs = 500  # Sampling rate in Hz
    duration = 3.0  # 3 seconds of data
    
    # 1. Generate Signal
    t, clean_ecg, noisy_ecg = generate_ecg(bpm, noise_amp, fs, duration)
    
    # 2. Filter Signal
    filtered_ecg = apply_digital_filter(noisy_ecg, fc, fs)
    
    # 3. FFT Analysis
    f_noisy, Y_noisy = compute_fft(noisy_ecg, fs)
    f_filt, Y_filt = compute_fft(filtered_ecg, fs)
    
    # 4. Filter Transfer Function & Poles (Continuous Domain)
    b_c, a_c = design_lowpass(fc)
    sys_cont = signal.lti(b_c, a_c)
    poles = sys_cont.poles
    
    # Update Equations Text
    wc = 2 * np.pi * fc
    eq_text = (
        f"Input Signal: Heart Rate = {bpm:.0f} BPM | Noise Amp = {noise_amp:.2f}\n"
        f"Filter Type: 2nd-Order Analog Butterworth Low-Pass (fc = {fc:.1f} Hz)\n"
        f"Transfer Function H(s) = ωc² / (s² + √2·ωc·s + ωc²)  [ωc = {wc:.1f} rad/s]\n"
        f"Poles: s1,2 = {poles[0].real:.2f} ± j{abs(poles[0].imag):.2f}"
    )
    lbl_equations.config(text=eq_text)
    
    # Re-draw Plots
    ax_time.clear()
    ax_fft.clear()
    ax_splane.clear()
    
    # Time Domain Plot
    ax_time.plot(t, noisy_ecg, color='lightgray', label='Noisy ECG (w/ 50Hz)')
    ax_time.plot(t, filtered_ecg, color='red', linewidth=1.5, label='Filtered ECG')
    ax_time.set_title("Time Domain: Heart ECG Signal")
    ax_time.set_xlabel("Time (s)")
    ax_time.set_ylabel("Amplitude")
    ax_time.legend(loc="upper right", fontsize=8)
    ax_time.grid(True, linestyle=':', alpha=0.7)
    
    # FFT Plot
    ax_fft.plot(f_noisy, Y_noisy, color='gray', alpha=0.6, label='Noisy Spectrum')
    ax_fft.plot(f_filt, Y_filt, color='blue', alpha=0.8, label='Filtered Spectrum')
    ax_fft.set_xlim(0, 100) # focus on 0-100Hz
    ax_fft.set_ylim(0, max(np.max(Y_noisy[1:]), 0.1) * 1.1)
    ax_fft.set_title("Frequency Domain: FFT Spectrum")
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Magnitude |X(f)|")
    ax_fft.axvline(50, color='orange', linestyle='--', linewidth=1, label='50 Hz Mains')
    ax_fft.legend(loc="upper right", fontsize=8)
    ax_fft.grid(True, linestyle=':', alpha=0.7)
    
    # S-Plane Plot (Poles)
    ax_splane.axvline(0, color='black', linewidth=1)
    ax_splane.axhline(0, color='black', linewidth=1)
    ax_splane.fill_betweenx([-wc*2, wc*2], -wc*2, 0, color='#e8f5e9', alpha=0.5, label="Stable (LHP)")
    ax_splane.scatter(poles.real, poles.imag, color='red', marker='x', s=100, linewidth=2, zorder=5)
    ax_splane.set_xlim(-wc*1.5, wc*0.5)
    ax_splane.set_ylim(-wc*1.5, wc*1.5)
    ax_splane.set_title("Laplace Domain: S-Plane Poles")
    ax_splane.set_xlabel("Real (σ)")
    ax_splane.set_ylabel("Imag (jω)")
    ax_splane.legend(loc="upper left", fontsize=8)
    ax_splane.grid(True, linestyle=':', alpha=0.7)
    
    fig.tight_layout()
    canvas.draw()

# --- MAIN WINDOW SETUP ---
root = tk.Tk()
root.title("Part 4 Original Simulation: ECG Fourier Analysis & Filter Design")
root.geometry("1200x800")

ctrl_frame = ttk.LabelFrame(root, text=" ECG & Filter Settings ")
ctrl_frame.pack(fill="x", padx=10, pady=5)

# Sliders
f_bpm = ttk.Frame(ctrl_frame)
f_bpm.pack(side="left", padx=15, pady=5)
ttk.Label(f_bpm, text="Heart Rate (BPM):").pack()
lbl_val_bpm = ttk.Label(f_bpm, text="75 BPM", font=("Arial", 9, "bold"), foreground="blue")
lbl_val_bpm.pack()
slider_bpm = ttk.Scale(f_bpm, from_=40, to=180, value=75, command=update_simulation)
slider_bpm.pack()

f_noise = ttk.Frame(ctrl_frame)
f_noise.pack(side="left", padx=15, pady=5)
ttk.Label(f_noise, text="50Hz Noise Amplitude:").pack()
lbl_val_noise = ttk.Label(f_noise, text="0.50", font=("Arial", 9, "bold"), foreground="red")
lbl_val_noise.pack()
slider_noise = ttk.Scale(f_noise, from_=0.0, to=2.0, value=0.5, command=update_simulation)
slider_noise.pack()

f_fc = ttk.Frame(ctrl_frame)
f_fc.pack(side="left", padx=15, pady=5)
ttk.Label(f_fc, text="Filter Cutoff (fc Hz):").pack()
lbl_val_fc = ttk.Label(f_fc, text="20.0 Hz", font=("Arial", 9, "bold"), foreground="green")
lbl_val_fc.pack()
slider_fc = ttk.Scale(f_fc, from_=5.0, to=80.0, value=20.0, command=update_simulation)
slider_fc.pack()

eq_frame = tk.LabelFrame(root, text=" Live Laplace Mathematical Model & Equations ", font=("Consolas", 10, "bold"))
eq_frame.pack(fill="x", padx=10, pady=5)
lbl_equations = tk.Label(eq_frame, text="", font=("Consolas", 10, "bold"), justify="left", fg="#1d3557")
lbl_equations.pack(anchor="w", padx=10, pady=5)

fig = plt.figure(figsize=(12, 6))
ax_time = fig.add_subplot(2, 1, 1)
ax_fft = fig.add_subplot(2, 2, 3)
ax_splane = fig.add_subplot(2, 2, 4)

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=5)

# Initialize
update_simulation()
root.mainloop()
