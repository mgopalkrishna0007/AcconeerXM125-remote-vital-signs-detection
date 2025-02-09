# Acconeer XM125 Remote Vital Signs Detection signal processing flow , Python API. 

This repository contains signal processing scripts for remote vital signs detection using the Acconeer XM125 radar sensor. The scripts are designed to process IQ data from the radar to extract breathing and heart rate information.

## Acconeer PCR Technology Overview

The Acconeer Pulsed Coherent Radar (PCR) technology enables the detection of minute movements by transmitting short radar pulses and coherently processing the received signals. [See Acconeer Documentation for further details](https://docs.acconeer.com/en/latest/pcr_tech/overview.html)

![image](https://github.com/user-attachments/assets/2eeb1647-2b67-4753-99aa-24dfd102621e)


## Signal Processing Flow

The signal processing flow is outlined below:

1.  **IQ Raw Data Acquisition:** The radar collects IQ radar sweeps, targeting the chest of a person.
2.  **Range Downsampling**
3.  **Low-Pass Filtering**
3.  **Phase Signal Extraction**: Three methods namely Variation trend method, DACM  and arctangent demodulation have been implemented. 
4.  **Signal filtering**: This includes baseline drift correction, clutter removal, bandpass filtering, and median filtering. Several scripts in this repository implement these filtering methods.
5.  **Waveform Reconstruction**: The signal is decomposed using Wavelet Transform (Symlet6, db4) and Empirical Mode Decomposition (EEMD) and reconstructed. Wavelet transform was explored as a separate signal processing step, but did not yield improved results in this implementation.
6.  **Rate Estimation:** Peak detection and FFT are used to estimate the breathing and heart rates.

A visual representation of the signal processing flow is shown below:

![image](https://github.com/user-attachments/assets/ed51c409-05d4-4f3e-88b6-7828aeb6316d)

## Variation Trend Method

![image](https://github.com/user-attachments/assets/1d142de6-003f-410f-9f08-14927df47950)
![image](https://github.com/user-attachments/assets/cd3ce379-b289-4dcc-ab57-c815c9251544)

## Empirical Mode Decomposition (EEMD)

Empirical Mode Decomposition (EEMD) is employed to decompose the phase signal into Intrinsic Mode Functions (IMFs). EEMD is a noise-assisted data analysis method. It decomposes a signal into a finite number of IMFs, which represent different frequency scales present in the data. In this project, EEMD is used to isolate the respiration component of the phase signal. The relevant IMFs are summed to reconstruct the respiration signal, which is then used for breath rate estimation.

The provided script extracts the respiration signal by summing the IMFs 3-5.
'
![IMG-20250207-WA0022](https://github.com/user-attachments/assets/61ce5d14-90d0-4435-8dd6-9efa5bbdb89d)

the output recosntructed signal looks like - 

![IMG-20250207-WA0021](https://github.com/user-attachments/assets/66fcf000-caef-447a-852c-e12df67cc407)


## Dependencies

*   Python 3.x
*   NumPy
*   SciPy
*   Matplotlib
*   PyEMD
*   h5py

