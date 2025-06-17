# Audio Spoof Detection using MFCC, CQCC, and Group Delay with Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Libraries](https://img.shields.io/badge/Librosa-TensorFlow-orange)

## 🔍 Overview

This project presents a novel deep learning approach to detect audio spoofing attacks in voice-controlled devices by combining three powerful audio features: 
- **Mel-Frequency Cepstral Coefficients (MFCC)**
- **Constant-Q Cepstral Coefficients (CQCC)**
- **Group Delay Cepstral Coefficients (GDCC)**

These features are used in conjunction with deep learning classifiers, namely **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory Networks (LSTM)**, to enhance Automatic Speaker Verification (ASV) systems against spoofing threats.

## 🎯 Objective

To build a robust and computationally efficient audio spoof detection system that:
- Extracts informative spectral and phase-based features.
- Combines cepstral features with group delay for better detection.
- Leverages CNN and LSTM models to classify bonafide vs spoofed audio.
- Evaluates performance using the DECRO (English) dataset.


## 🧠 Key Features

- ✅ **Feature Extraction**:
  - MFCC and CQCC highlight spectral information.
  - Group Delay captures phase-related characteristics.

- 🧮 **Deep Learning Models**:
  - CNN captures local patterns in audio signals.
  - LSTM captures temporal dependencies over time.

- 📈 **Performance Metrics**:
  - Achieved up to **99.97% accuracy** with **LSTM + MFCC + GDCC**.
  - Evaluated using Precision, Recall, F1-Score, EER, and t-DCF.


