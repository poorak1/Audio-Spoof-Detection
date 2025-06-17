import os
import csv
import numpy as np
import librosa
from scipy.fftpack import dct

folder_path = r"D:\VLD\Dataset\DECRO-eng\training\spoof"
testing_bonafide = [f for f in os.listdir(folder_path)]
file_num=0
n_cqcc=13
for file in testing_bonafide:
    temp_path=os.path.join(folder_path, file)
    signal, sr = librosa.load(temp_path)
    CQT = np.abs(librosa.cqt(signal, sr=sr, hop_length=512))
    log_CQT = np.log(CQT)  
    cqcc = dct(log_CQT)
    cqcc = cqcc[:n_cqcc]
    mean_cqcc = np.mean(np.abs(cqcc), axis=1)
    #std_cqcc = np.std(np.abs(cqcc), axis=1)
    #features = np.concatenate([mean_cqcc, std_cqcc])
    #features=features.flatten()
    with open('cqcc_training_spoof.csv', 'a', newline='') as newfile:
        writer = csv.writer(newfile)
        writer.writerow(mean_cqcc)
    print(f"Extracted Features for File Number: {file_num+1}")
    file_num+=1




