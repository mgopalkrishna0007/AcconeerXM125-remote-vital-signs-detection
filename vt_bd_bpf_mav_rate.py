# this code applies baseline drift on the phase signal plot and then applies bandpass filter 
# then after application of bandpass filter it applies moving average filter and then finally applies fft for bpm calculation
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, detrend

# File path
file_path = r"/home/sarthak/breathing_files/amritoutdoor/Laying Vertical/amrit05layingvL.h5"
# Load data
with h5py.File(file_path, "r") as f:
    frame = f["sessions/session_0/group_0/entry_0/result/frame"]
    real_part = np.array(frame["real"], dtype=np.float64)  # Extract real part
    imag_part = np.array(frame["imag"], dtype=np.float64)  # Extract imaginary part

# Combine real and imaginary parts into complex IQ data
IQ_data = real_part + 1j * imag_part  # Shape: (1794, 32, 40)
IQ_data = IQ_data.transpose(2, 1, 0)  # Transpose to (antennas, range bins, sweeps)

# Parameters
fs = 10  # Sampling frequency (Hz)

# Compute magnitude of IQ data
magnitude_data = np.abs(IQ_data)

# Step 1: Baseline Drift Correction (Detrending)
detrended_data = np.zeros_like(magnitude_data)
for i in range(magnitude_data.shape[0]):  # Antennas
    for j in range(magnitude_data.shape[1]):  # Range bins
        detrended_data[i, j, :] = detrend(magnitude_data[i, j, :], type='linear')

# Bandpass Filter Function
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, data)

# Step 2: Apply Bandpass Filter to Detrended Data
bandpass_filtered_data = np.zeros_like(detrended_data)
for i in range(detrended_data.shape[0]):  # Antennas
    for j in range(detrended_data.shape[1]):  # Range bins
        bandpass_filtered_data[i, j, :] = bandpass_filter(
            detrended_data[i, j, :], 
            lowcut=0.1,  # Lower cutoff frequency (0.1 Hz)
            highcut=0.5,  # Upper cutoff frequency (0.5 Hz)
            fs=fs
        )

# Moving Average Filter Function
def moving_average_filter(data, window_size):
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size
    pad_width = window_size // 2
    padded_data = np.pad(data, pad_width, mode='reflect')
    return np.convolve(padded_data, np.ones(window_size) / window_size, mode='valid')

# Step 3: Apply Moving Average Filter to Bandpass-Filtered Data
moving_avg_filtered_data = np.zeros_like(bandpass_filtered_data)
for i in range(bandpass_filtered_data.shape[0]):  # Antennas
    for j in range(bandpass_filtered_data.shape[1]):  # Range bins
        moving_avg_filtered_data[i, j, :] = moving_average_filter(
            bandpass_filtered_data[i, j, :], 
            window_size=int(fs * 2)  # 2-second window
        )

# Step 4: Perform FFT on Moving Average Filtered Data
fft_results = np.abs(np.fft.fft(moving_avg_filtered_data, axis=2))
freqs = np.fft.fftfreq(moving_avg_filtered_data.shape[2], d=1/fs)

# Breath Rate Calculation
breath_rates_bpm = []
for i in range(moving_avg_filtered_data.shape[0]):  # Antennas
    for j in range(moving_avg_filtered_data.shape[1]):  # Range bins
        positive_freqs = freqs[freqs >= 0]
        fft_magnitude = fft_results[i, j, freqs >= 0]
        
        # Focus on respiratory frequency range (0.1 - 0.5 Hz)
        respiratory_mask = (positive_freqs >= 0.1) & (positive_freqs <= 0.5)
        resp_freqs = positive_freqs[respiratory_mask]
        resp_magnitudes = fft_magnitude[respiratory_mask]
        
        if len(resp_magnitudes) > 0:
            peak_index = np.argmax(resp_magnitudes)
            peak_freq_hz = resp_freqs[peak_index]
            if peak_freq_hz > 0:
                breath_rate_bpm = peak_freq_hz * 60  # Convert Hz to BPM
                breath_rates_bpm.append(breath_rate_bpm)

# Average Breath Rate Calculation
if breath_rates_bpm:
    average_breath_rate_bpm = np.mean(breath_rates_bpm)
    print(f"Estimated Average Breath Rate: {average_breath_rate_bpm:.2f} BPM")
else:
    print("No valid breath rate detected.")

# Plotting Results
plt.figure(figsize=(15, 10))

# Detrended Signal
plt.subplot(3, 1, 1)
plt.plot(detrended_data[0, 0, :], label='Detrended Signal', color='blue')
plt.title('Detrended Signal (Baseline Drift Removed)')
plt.xlabel('Frame Index')
plt.ylabel('Amplitude')
plt.grid(True)

# Bandpass Filtered Signal
plt.subplot(3, 1, 2)
plt.plot(bandpass_filtered_data[0, 0, :], label='Bandpass Filtered', color='orange')
plt.title('Bandpass Filtered Signal (0.1–0.5 Hz)')
plt.xlabel('Frame Index')
plt.ylabel('Amplitude')
plt.grid(True)

# Moving Average Filtered Signal
plt.subplot(3, 1, 3)
plt.plot(moving_avg_filtered_data[0, 0, :], label='Moving Average Filtered', color='green')
plt.title('Moving Average Filtered Signal')
plt.xlabel('Frame Index')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot FFT Magnitude Spectrum
plt.figure(figsize=(12, 6))
positive_freqs = freqs[freqs >= 0]
plt.plot(positive_freqs, fft_results[0, 0, freqs >= 0], label='FFT Magnitude', color='purple')
plt.title('FFT Magnitude Spectrum (0.1–0.5 Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.show()