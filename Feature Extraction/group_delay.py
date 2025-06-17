import os
import librosa
import numpy as np
import csv

folder_path = r"D:\VLD\Dataset\DECRO-eng\training\spoof"
testing_bonafide = [f for f in os.listdir(folder_path)]
file_num = 0

for file in testing_bonafide:
    temp_path = os.path.join(folder_path, file)
    signal, sr = librosa.load(temp_path)
    scale = librosa.stft(signal)
    magnitude_stft = np.abs(scale)
    
    n = np.arange(len(signal))
    modified_signal = np.multiply(n, signal)
    modified_stft = librosa.stft(modified_signal)
    
    X_r = np.real(scale)
    X_i = np.imag(scale)
    Y_r = np.real(modified_stft)
    Y_i = np.imag(modified_stft)
    
    RHO = 0.6
    GAMMA = 0.3
    nc = 13
    
    log_magnitude = np.log(magnitude_stft + 1e-10)
    real_cepstrum = np.fft.ifft(log_magnitude, axis=0).real
    lifter = np.ones_like(real_cepstrum)
    lifter[nc + 1:] = 0
    lifter[nc] = 0.5
    smoothed_cepstrum = real_cepstrum * lifter
    smoothed_log_magnitude = np.fft.fft(smoothed_cepstrum, axis=0).real
    smoothed_magnitude = np.exp(smoothed_log_magnitude)
    
    gd = (X_r * Y_r + X_i * Y_i) / (smoothed_magnitude)**(2*RHO)
    mgd = (gd * ((np.abs(gd))**GAMMA)) / np.abs(gd)
    mgdcc = np.fft.fft(mgd)
    
    mean_mgdcc = np.mean(np.abs(mgdcc[:13, :]), axis=1)
    std_mgdcc = np.std(np.abs(mgdcc[:13, :]), axis=1)
    features = np.concatenate([mean_mgdcc, std_mgdcc])
    
    with open('mgdcc_training_spoof.csv', 'a', newline='') as newfile:
        writer = csv.writer(newfile)
        writer.writerow(features)
    
    print(f"Extracted Features for File Number: {file_num+1}")
    file_num += 1