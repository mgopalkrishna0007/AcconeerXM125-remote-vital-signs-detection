# #this code uses the phase signal plot to further apply baseline drift on it 
# # it then applies bandpass filter and then performs fft to determine the bpm 
# import numpy as np
# import h5py
# import matplotlib.pyplot as plt
# from scipy.signal import detrend, butter, sosfilt

# # File path
# file_path = r"/home/sarthak/breathing_files/amritoutdoor/Moving/amrit04movingL.h5"

# # Load data
# with h5py.File(file_path, "r") as f:
#     frame = f["sessions/session_0/group_0/entry_0/result/frame"]
#     real_part = np.array(frame["real"], dtype=np.float64)  # Extract real part
#     imag_part = np.array(frame["imag"], dtype=np.float64)  # Extract imaginary part

# # Combine real and imaginary parts into complex IQ data
# IQ_data = real_part + 1j * imag_part  # Shape: (1794, 32, 40)

# # Transpose data to match MATLAB's order: (antennas x range bins x sweeps)
# IQ_data = IQ_data.transpose(2, 1, 0)  # Shape: (40, 32, 1794)

# # Parameters
# fs = 100  # Sweep rate (Hz)
# D = 100  # Downsampling factor

# # Compute the magnitude of IQ data (sweeps x range bins)
# magnitude_data = np.abs(IQ_data)

# # Find the range bin with the highest peak magnitude (across all sweeps)
# mean_magnitude = np.mean(magnitude_data, axis=2)  # Mean over sweeps
# peak_range_index = np.argmax(mean_magnitude, axis=1)  # Index for each antenna

# # Select the range indices based on the peak range bin
# range_start_bin = max(0, peak_range_index[0] - 5)  # Adjust as needed
# range_end_bin = min(IQ_data.shape[1], peak_range_index[0] + 5)
# range_indices = np.arange(range_start_bin, range_end_bin + 1)

# # Downsampling
# downsampled_data = IQ_data[:, range_indices[::D], :]  # Shape: (40, downsampled ranges, 1794)

# # Temporal low-pass filter parameters
# tau_iq = 0.5  # Time constant for low-pass filter (seconds)
# alpha_iq = np.exp(-2 / (tau_iq * fs))  # Low-pass filter coefficient

# # Initialize filtered data
# filtered_data = np.zeros_like(downsampled_data)
# filtered_data[:, :, 0] = downsampled_data[:, :, 0]

# # Apply temporal low-pass filter
# for s in range(1, downsampled_data.shape[2]):
#     filtered_data[:, :, s] = alpha_iq * filtered_data[:, :, s - 1] + \
#                              (1 - alpha_iq) * downsampled_data[:, :, s]

# # Baseline drift correction using detrending
# corrected_data = np.zeros_like(filtered_data)

# for i in range(corrected_data.shape[0]):  # For each antenna
#     for j in range(corrected_data.shape[1]):  # For each range bin
#         corrected_data[i, j, :] = detrend(filtered_data[i, j, :], type='linear')

# # Bandpass filter function
# def butter_bandpass(lowcut, highcut, fs, order=6):
#     nyq = 0.6 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     sos = butter(order, [low, high], btype='band', output='sos')
#     return sos

# def bandpass_filter(data, lowcut, highcut, fs, order=6):
#     sos = butter_bandpass(lowcut, highcut, fs, order=order)
#     filtered_data = sosfilt(sos, data)
#     return filtered_data

# # Apply bandpass filter for breathing rate
# bandpass_filtered_data = np.zeros_like(corrected_data)

# for i in range(corrected_data.shape[0]):  # For each antenna
#     for j in range(corrected_data.shape[1]):  # For each range bin
#         bandpass_filtered_data[i, j, :] = bandpass_filter(
#             corrected_data[i, j, :], 
#             lowcut=0.1,  # Lower bound of breathing frequency (6 breaths/min)
#             highcut=0.3,  # Upper bound of breathing frequency (30 breaths/min)
#             fs=fs
#         )

# # Perform FFT on bandpass filtered data
# fft_results = np.abs(np.fft.fft(bandpass_filtered_data, axis=2))
# freqs = np.fft.fftfreq(bandpass_filtered_data.shape[2], d=1/fs)

# # Breath rate calculation
# breath_rates_bpm = []

# for i in range(bandpass_filtered_data.shape[0]):  # For each antenna
#     for j in range(bandpass_filtered_data.shape[1]):  # For each range bin
#         positive_freqs = freqs[freqs >= 0]
#         fft_magnitude = fft_results[i, j, freqs >= 0]
        
#         if len(fft_magnitude) > 0:
#             peak_index = np.argmax(fft_magnitude)
#             peak_freq_hz = positive_freqs[peak_index]
#             if peak_freq_hz > 0:  # Only consider positive frequencies for BPM calculation
#                 breath_rate_bpm = peak_freq_hz * 60  # Convert Hz to BPM
#                 breath_rates_bpm.append(breath_rate_bpm)

# # Average breath rate calculation
# if breath_rates_bpm:
#     average_breath_rate_bpm = np.mean(breath_rates_bpm)
#     print(f"Estimated Average Breath Rate: {average_breath_rate_bpm:.2f} BPM")
# else:
#     print("No valid breath rate detected.")

# # Plotting bandpass filtered signal
# plt.figure(figsize=(12, 6))
# plt.plot(range(len(bandpass_filtered_data[0,0,:])), bandpass_filtered_data[0,0,:], linewidth=1.5)
# plt.xlabel('Frame Index (sweeps)')
# plt.ylabel('Bandpass Filtered Signal Amplitude')
# plt.title('Bandpass Filtered Signal vs. Frames')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, sosfilt

# File path
file_path = r"/home/sarthak/dhama_breathing_files/sitting/dhama01sitting.h5"

# Load data
with h5py.File(file_path, "r") as f:
    frame = f["sessions/session_0/group_0/entry_0/result/frame"]
    real_part = np.array(frame["real"], dtype=np.float64)  # Extract real part
    imag_part = np.array(frame["imag"], dtype=np.float64)  # Extract imaginary part

# Combine real and imaginary parts into complex IQ data
IQ_data = real_part + 1j * imag_part  # Shape: (1794, 32, 40)

# Transpose data to match MATLAB's order: (antennas x range bins x sweeps)
IQ_data = IQ_data.transpose(2, 1, 0)  # Shape: (40, 32, 1794)

# Parameters
fs = 10  # Sweep rate (Hz)
D = 100  # Downsampling factor

# Compute the magnitude of IQ data (sweeps x range bins)
magnitude_data = np.abs(IQ_data)

# Find the range bin with the highest peak magnitude (across all sweeps)
mean_magnitude = np.mean(magnitude_data, axis=2)  # Mean over sweeps
peak_range_index = np.argmax(mean_magnitude, axis=1)  # Index for each antenna

# Select the range indices based on the peak range bin
range_start_bin = max(0, peak_range_index[0] - 5)  # Adjust as needed
range_end_bin = min(IQ_data.shape[1], peak_range_index[0] + 5)
range_indices = np.arange(range_start_bin, range_end_bin + 1)

# Downsampling
downsampled_data = IQ_data[:, range_indices[::D], :]  # Shape: (40, downsampled ranges, 1794)

# Temporal low-pass filter parameters
tau_iq = 0.5  # Time constant for low-pass filter (seconds)
alpha_iq = np.exp(-2 / (tau_iq * fs))  # Low-pass filter coefficient

# Initialize filtered data
filtered_data = np.zeros_like(downsampled_data)
filtered_data[:, :, 0] = downsampled_data[:, :, 0]

# Apply temporal low-pass filter
for s in range(1, downsampled_data.shape[2]):
    filtered_data[:, :, s] = alpha_iq * filtered_data[:, :, s - 1] + \
                             (1 - alpha_iq) * downsampled_data[:, :, s]

# Baseline drift correction using detrending
corrected_data = np.zeros_like(filtered_data)

for i in range(corrected_data.shape[0]):  # For each antenna
    for j in range(corrected_data.shape[1]):  # For each range bin
        corrected_data[i, j, :] = detrend(filtered_data[i, j, :], type='linear')

# Bandpass filter function
def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def bandpass_filter(data, lowcut, highcut, fs, order=6):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = sosfilt(sos, data)
    return filtered_data

# Apply bandpass filter for breathing rate (0.1–0.4 Hz)
bandpass_filtered_data = np.zeros_like(corrected_data)

for i in range(corrected_data.shape[0]):  # For each antenna
    for j in range(corrected_data.shape[1]):  # For each range bin
        bandpass_filtered_data[i, j, :] = bandpass_filter(
            corrected_data[i, j, :], 
            lowcut=0.1,  # Lower bound of breathing frequency (0.1 Hz)
            highcut=0.4,  # Upper bound of breathing frequency (0.4 Hz)
            fs=fs
        )

# Perform FFT on bandpass filtered data
fft_results = np.abs(np.fft.fft(bandpass_filtered_data, axis=2))
freqs = np.fft.fftfreq(bandpass_filtered_data.shape[2], d=1/fs)

# Breath rate calculation in the 0.1–0.4 Hz range
breath_rates_bpm = []

for i in range(bandpass_filtered_data.shape[0]):  # For each antenna
    for j in range(bandpass_filtered_data.shape[1]):  # For each range bin
        # Mask for the frequency range of interest (0.1–0.4 Hz)
        freq_mask = (freqs >= 0.1) & (freqs <= 0.4)
        positive_freqs = freqs[freq_mask]
        fft_magnitude = fft_results[i, j, freq_mask]
        
        if len(fft_magnitude) > 0:
            peak_index = np.argmax(fft_magnitude)
            peak_freq_hz = positive_freqs[peak_index]
            if peak_freq_hz > 0:  # Only consider positive frequencies for BPM calculation
                breath_rate_bpm = peak_freq_hz * 60  # Convert Hz to BPM
                breath_rates_bpm.append(breath_rate_bpm)

# Average breath rate calculation
if breath_rates_bpm:
    average_breath_rate_bpm = np.mean(breath_rates_bpm)
    print(f"Estimated Average Breath Rate (0.1–0.4 Hz): {average_breath_rate_bpm:.2f} BPM")
else:
    print("No valid breath rate detected in the 0.1–0.4 Hz range.")

# Plotting bandpass filtered signal
plt.figure(figsize=(12, 6))
plt.plot(range(len(bandpass_filtered_data[0, 0, :])), bandpass_filtered_data[0, 0, :], linewidth=1.5)
plt.xlabel('Frame Index (sweeps)')
plt.ylabel('Bandpass Filtered Signal Amplitude')
plt.title('Bandpass Filtered Signal (0.1–0.4 Hz) vs. Frames')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting FFT spectrum in the 0.1–0.4 Hz range
plt.figure(figsize=(12, 6))
for i in range(bandpass_filtered_data.shape[0]):  # For each antenna
    for j in range(bandpass_filtered_data.shape[1]):  # For each range bin
        freq_mask = (freqs >= 0.1) & (freqs <= 0.4)
        positive_freqs = freqs[freq_mask]
        fft_magnitude = fft_results[i, j, freq_mask]
        plt.plot(positive_freqs, fft_magnitude, label=f"Antenna {i}, Range {j}")

plt.xlabel('Frequency (Hz)')
plt.ylabel('FFT Magnitude')
plt.title('FFT Spectrum (0.1–0.4 Hz)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()