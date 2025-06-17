import os
import librosa
import numpy as np
import csv

folder_path = r"D:\VLD\Dataset\DECRO-eng\training\spoof"
testing_bonafide = [f for f in os.listdir(folder_path)]
file_num = 0

n_ceps=13

for file in testing_bonafide:
    temp_path = os.path.join(folder_path, file)
    signal, sr = librosa.load(temp_path)
    stft = librosa.stft(signal)
    phase = np.angle(stft)
    
    phase_unwrapped = np.unwrap(phase, axis=0)
    group_delay = -np.diff(phase_unwrapped, axis=0)
    group_delay = np.pad(group_delay, ((1, 0), (0, 0)), mode='constant')
    log_spectrum = np.log(np.abs(group_delay) + 1e-10)
    gdcc = np.fft.ifft(log_spectrum, axis=0).real
    gdcc = gdcc[:n_ceps]
    
    mean_mgdcc = np.mean(np.abs(gdcc), axis=1)
    #std_mgdcc = np.std(np.abs(gdcc), axis=1)
    #features = np.concatenate([mean_mgdcc, std_mgdcc])
    
    with open('gdcc_training_spoof.csv', 'a', newline='') as newfile:
        writer = csv.writer(newfile)
        writer.writerow(mean_mgdcc)
    
    print(f"Extracted Features for File Number: {file_num+1}")
    file_num += 1