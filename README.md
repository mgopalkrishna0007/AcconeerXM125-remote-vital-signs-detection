# Acconeer XM125 Remote Vital Signs Detection signal processing flow , Python API. 

This repository contains signal processing scripts for remote vital signs detection using the Acconeer XM125 radar sensor. The scripts are designed to process IQ data from the radar to extract breathing and heart rate information.

## Acconeer PCR Technology Overview

The Acconeer Pulsed Coherent Radar (PCR) technology enables the detection of minute movements by transmitting short radar pulses and coherently processing the received signals. [See Acconeer Documentation for further details](https://docs.acconeer.com/en/latest/pcr_tech/overview.html)

![image](https://github.com/user-attachments/assets/2eeb1647-2b67-4753-99aa-24dfd102621e)


## Signal Processing Flow

The signal processing flow is outlined below:

1.  **IQ Raw Data Acquisition:** The radar collects IQ radar sweeps, targeting the chest of a person.
2.  **Motion Signal Extraction**: This involves range downsampling, low-pass filtering, and phase unwrapping.
3.  **Phase Signal Extraction**: Three methods namely Variation trend method, DACM (Differential Arc Cosine Method) and arctangent demodulation have been implemented.
4.  **Signal Conditioning**: This includes baseline drift correction, clutter removal, bandpass filtering, and median filtering. Several scripts in this repository implement these filtering methods.
5.  **Waveform Reconstruction**: The signal is decomposed using Wavelet Transform (Symlet6, db4) and Empirical Mode Decomposition (EEMD) and reconstructed to improve the signal quality. Wavelet transform was explored as a separate signal processing step, but did not yield improved results in this implementation.
6.  **Rate Estimation:** Peak detection and FFT are used to estimate the breathing and heart rates.

A visual representation of the signal processing flow is shown below:

![image](https://github.com/user-attachments/assets/ed51c409-05d4-4f3e-88b6-7828aeb6316d)

![image](https://github.com/user-attachments/assets/e286843e-9e72-4777-b4c1-c65fbcd8c1d0)


## Motion Signal Extraction (Variation Trend Method)

![image](https://github.com/user-attachments/assets/1d142de6-003f-410f-9f08-14927df47950)


The following steps are used to obtain the motion signal from the IQ data:

1.  **Data Acquisition**:
    The radar collects IQ radar sweeps with a range covering the chest of a breathing person. The data samples from the IQ API are represented by

    $S(r,n) = I(r,n) + jQ(r,n)$

    where \( r \) is the range index and \( n \) is the sweep index.

2.  **Range Downsampling**:
    The IQ samples are downsampled in range to reduce the amount of data for further processing:

    $S_d(i,n) = S(i \cdot D, n)$

    where \( D \) is a range downsampling factor.

3.  **Low-Pass Filtering**:
    A noise-reducing low-pass filter is applied in the time (sweep) dimension:

    $S_f(i,n) = \alpha \cdot S_f(i, n-1) + (1 - \alpha) \cdot S_d(i,n)$

    where \( \alpha \) is the filter coefficient.

4.  **Phase Unwrapping**:
    The phase of the IQ samples is unwrapped:

    $\Phi(n) = \text{angle}\left(\overline{\text{mean}(S_f(:,n-1))} \cdot \text{mean}(S_f(:,n))\right) + \beta \cdot \Phi(n-1)$

    where \( \beta \) is a high-pass filter factor, and \( S_f \) is the low-pass filtered signal.

    ![image](https://github.com/user-attachments/assets/cd3ce379-b289-4dcc-ab57-c815c9251544)

## Signal Conditioning - Filtering Methods

This repository contains scripts that employ various filtering methods to enhance the signal quality:

*   **Baseline Drift Correction:** This involves removing the slow-varying trends or baseline wander in the phase signal. Baseline drift can arise from various factors, such as sensor movement or environmental changes. The goal is to stabilize the signal and remove low-frequency artifacts, allowing for better detection of the respiratory signal.
     *Method*: High pass filter is applied or, baseline is estimated using a moving median filter with a large window size (e.g., 2 seconds) and then subtracting this baseline from the original signal.

*   **Moving Average Filter**: This filter smooths the signal by calculating the average of data points over a defined window. It reduces noise and sharp fluctuations, making it easier to identify underlying trends, such as the average respiration rate over a short period.

*   **Median Filtering**: A median filter is applied to the phase signal to remove baseline drift and noise. This involves calculating a moving median over a specified window size and replacing each data point with the median value of its neighbors.

*   **Clutter Removal**: This step involves subtracting the mean phase to remove stationary components or background clutter from the signal. This is essential for highlighting dynamic changes related to breathing.

*   **Bandpass Filtering**: A bandpass filter is applied to isolate the respiration component of the phase signal. Typical cutoff frequencies are 0.1 Hz to 0.5 Hz, corresponding to normal breathing rates. `butterworth` filter is implemented

![image](https://github.com/user-attachments/assets/ca7f13bc-fdd2-4c5d-ba6f-2dcc381d852a)


## Empirical Mode Decomposition (EEMD)

Empirical Mode Decomposition (EEMD) is employed to decompose the phase signal into Intrinsic Mode Functions (IMFs). EEMD is a noise-assisted data analysis method. It decomposes a signal into a finite number of IMFs, which represent different frequency scales present in the data. In this project, EEMD is used to isolate the respiration component of the phase signal. The relevant IMFs are summed to reconstruct the respiration signal, which is then used for breath rate estimation.

The provided script extracts the respiration signal by summing the IMFs 3-5.
'
![IMG-20250207-WA0022](https://github.com/user-attachments/assets/61ce5d14-90d0-4435-8dd6-9efa5bbdb89d)

the out put recosntructed signal looks like - 

![IMG-20250207-WA0021](https://github.com/user-attachments/assets/66fcf000-caef-447a-852c-e12df67cc407)


## Dependencies

*   Python 3.x
*   NumPy
*   SciPy
*   Matplotlib
*   PyEMD
*   h5py

