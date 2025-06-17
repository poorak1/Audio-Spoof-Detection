import librosa
import numpy as np

n_ceps=13

y, sr = librosa.load(r"D:\VLD\Dataset\DECRO-eng\testing\bonafide\LA_E_1003128.wav")

stft = librosa.stft(y)
phase = np.angle(stft)
phase_unwrapped = np.unwrap(phase, axis=0)
group_delay = -np.diff(phase_unwrapped, axis=0)
group_delay = np.pad(group_delay, ((1, 0), (0, 0)), mode='constant')
log_spectrum = np.log(np.abs(group_delay) + 1e-10)
gdcc = np.fft.ifft(log_spectrum, axis=0).real

gdcc = gdcc[:n_ceps]


