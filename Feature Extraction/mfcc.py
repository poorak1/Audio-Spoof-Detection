import librosa
import csv
import os
import numpy as np

folder_path = r"D:\VLD\Dataset\DECRO-eng\training\spoof"
testing_bonafide = [f for f in os.listdir(folder_path)]
file_num=0
for file in testing_bonafide:
    temp_path=os.path.join(folder_path, file)
    signal, sr = librosa.load(temp_path)
    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
    mean_mfcc = np.mean(np.abs(mfccs), axis=1)
    #std_mfcc = np.std(np.abs(mfccs), axis=1)
    #features = np.concatenate([mean_mfcc, std_mfcc])
    #features=features.flatten()
    features=mean_mfcc
    with open('mfcc_training_spoof.csv', 'a', newline='') as newfile:
        writer = csv.writer(newfile)
        writer.writerow(features)
    print(f"Extracted Features for File Number: {file_num+1}")
    file_num+=1

