# import numpy as np
# import matplotlib.pyplot as plt

# def generate_breathing_signal(duration=10, fs=1000, breath_rate=0.25):
#     """
#     Generate a synthetic breathing signal.
    
#     Parameters:
#     - duration: Duration of the signal in seconds
#     - fs: Sampling frequency in Hz
#     - breath_rate: Breaths per second (default is 0.25 for 15 breaths per minute)
    
#     Returns:
#     - t: Time vector
#     - signal: Breathing signal
#     """
#     t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # Time vector
#     # Generate a sinusoidal waveform to simulate breathing pattern
#     signal = 0.5 * (1 + np.sin(2 * np.pi * breath_rate * t))  # Normalize the amplitude between 0 and 1
#     return t, signal

# def generate_heartbeat_signal(duration=10, fs=1000, heart_rate=1.2):
#     """
#     Generate a synthetic heartbeat signal.
    
#     Parameters:
#     - duration: Duration of the signal in seconds
#     - fs: Sampling frequency in Hz
#     - heart_rate: Heartbeats per second (default is 1.2 for ~72 beats per minute)
    
#     Returns:
#     - t: Time vector
#     - signal: Heartbeat signal
#     """
#     t = np.linspace(0, duration, int(fs * duration), endpoint=False)  # Time vector
#     # Generate a square waveform to simulate heartbeat pattern
#     pulse_duration = 0.5 / heart_rate  # Duration of one heartbeat
#     pulse = np.where((t % (1 / heart_rate)) < pulse_duration, 1, 0)  # Square wave for heartbeat
#     return t, pulse

# def add_noise(signal, noise_level=0.05):
#     """
#     Add Gaussian noise to a signal.
    
#     Parameters:
#     - signal: Input signal (1D array)
#     - noise_level: Standard deviation of the Gaussian noise
    
#     Returns:
#     - noisy_signal: Signal with added noise
#     """
#     noise = np.random.normal(0, noise_level, len(signal))
#     noisy_signal = signal + noise
#     return noisy_signal

# # Example usage
# if __name__ == "__main__":
#     duration = 60  # Duration of the signals in seconds
#     fs = 1000      # Sampling frequency in Hz
#     breath_rate = 0.25  # Breathing rate in Hz (15 breaths per minute)
#     heart_rate = 1.2   # Heart rate in Hz (72 beats per minute)
    
#     # Generate signals
#     t_breathing, breathing_signal = generate_breathing_signal(duration, fs, breath_rate)
#     t_heartbeat, heartbeat_signal = generate_heartbeat_signal(duration, fs, heart_rate)
    
#     # Combine signals and add noise
#     combined_signal = breathing_signal + heartbeat_signal * 0.5  # Scale heartbeat for better visualization
#     noisy_combined_signal = add_noise(combined_signal)

#     # Plotting the signals
#     plt.figure(figsize=(12, 8))
    
#     plt.subplot(3, 1, 1)
#     plt.plot(t_breathing, breathing_signal, label='Breathing Signal', color='blue')
#     plt.title('Synthetic Breathing Signal')
#     plt.xlabel('Time [s]')
#     plt.ylabel('Amplitude')
    
#     plt.subplot(3, 1, 2)
#     plt.plot(t_heartbeat, heartbeat_signal * 0.5, label='Heartbeat Signal', color='red')  # Scale heartbeat for better visibility
#     plt.title('Synthetic Heartbeat Signal')
#     plt.xlabel('Time [s]')
#     plt.ylabel('Amplitude')
    
#     plt.subplot(3, 1, 3)
#     plt.plot(t_breathing, noisy_combined_signal, label='Noisy Combined Signal', color='green')
#     plt.title('Combined Breathing and Heartbeat Signal with Noise')
#     plt.xlabel('Time [s]')
#     plt.ylabel('Amplitude')
    
#     plt.tight_layout()
#     plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt

def bandpass_filter(signal, fs, lowcut=0.1, highcut=0.5):
    """
    Apply a bandpass filter to isolate the breathing frequency range.
    """
    sos = butter(4, [lowcut, highcut], btype='band', fs=fs, output='sos')
    filtered_signal = sosfilt(sos, signal)
    return filtered_signal

def calculate_breathing_rate(signal, fs):
    """
    Calculate breathing rate in bpm using FFT.
    """
    # Perform FFT
    n = len(signal)
    freqs = np.fft.fftfreq(n, d=1/fs)  # Frequency bins
    fft_values = np.fft.fft(signal)
    
    # Focus on positive frequencies
    positive_freqs = freqs[freqs >= 0]
    magnitude = np.abs(fft_values[freqs >= 0])
    
    # Focus on breathing frequency range (0.1 Hz to 0.5 Hz)
    valid_range = (positive_freqs >= 0.1) & (positive_freqs <= 0.5)
    valid_freqs = positive_freqs[valid_range]
    valid_magnitudes = magnitude[valid_range]
    
    # Find peak frequency
    if len(valid_magnitudes) > 0:
        peak_index = np.argmax(valid_magnitudes)
        peak_freq = valid_freqs[peak_index]
        breath_rate_bpm = peak_freq * 60
        return breath_rate_bpm
    else:
        return 0

# Example usage
if __name__ == "__main__":
    # Generate synthetic breathing signal
    duration = 60 # seconds
    fs = 1000      # Sampling frequency in Hz
    breath_rate_hz = 0.2  # Breathing rate in Hz (15 bpm)

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    breathing_signal = 0.5 * (1 + np.sin(2 * np.pi * breath_rate_hz * t))
    
    # Add noise
    noisy_signal = breathing_signal + np.random.normal(0, 0.05, len(breathing_signal))
    
    # Apply bandpass filter
    filtered_signal = bandpass_filter(noisy_signal, fs)

    # Calculate breathing rate
    estimated_breathing_rate = calculate_breathing_rate(filtered_signal, fs)
    
    print(f"Estimated Breathing Rate: {estimated_breathing_rate:.2f} bpm")
    
    # Plot signals and spectrum
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, noisy_signal)
    plt.title("Noisy Breathing Signal")
    
    plt.subplot(3, 1, 2)
    plt.plot(t, filtered_signal)
    plt.title("Filtered Breathing Signal")
    
    plt.subplot(3, 1, 3)
    freqs = np.fft.fftfreq(len(filtered_signal), d=1/fs)
    fft_values = np.abs(np.fft.fft(filtered_signal))
    
    plt.plot(freqs[freqs >= 0], fft_values[freqs >= 0])
    plt.title("FFT Spectrum")
    
    plt.tight_layout()
    plt.show()
