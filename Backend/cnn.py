import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_recall_fscore_support, roc_curve
import tensorflow as tf
from tensorflow.keras import layers, models
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
tf.experimental.numpy.random.seed(42)

def load_and_preprocess(file_path, label):
    data = pd.read_csv(file_path, header=None)
    X = data.values
    y = np.full(X.shape[0], label)
    return X, y

def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    eer = fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    return eer

def compute_tdcf_deepfake(y_true, y_scores, p_bonafide=0.95, c_miss_asv=1, c_fa_asv=10, c_miss_cm=1, c_fa_cm=10):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    p_spoof = 1 - p_bonafide
    
    p_miss_asv = 0.05
    p_fa_asv = 0.01
    
    #c0 = p_bonafide * c_miss_asv
    #c1 = p_spoof * c_fa_asv
    c_asv = p_bonafide * p_miss_asv * c_miss_asv + p_spoof * p_fa_asv * c_fa_asv
    
    tdcf_norm = []
    for i in range(len(thresholds)):
        p_miss_cm = fnr[i]
        p_fa_cm = fpr[i]
        c_cm = p_bonafide * p_miss_cm * c_miss_cm + p_spoof * p_fa_cm * c_fa_cm
        tdcf = c_cm + p_bonafide * (1 - p_miss_cm) * p_miss_asv * c_miss_asv + p_spoof * (1 - p_fa_cm) * p_fa_asv * c_fa_asv
        tdcf_norm.append(tdcf / c_asv)
    
    min_tdcf = min(tdcf_norm)
    return min_tdcf

X_bonafide, y_bonafide = load_and_preprocess('mfcc_training_bonafide.csv', 0)
X_spoof, y_spoof = load_and_preprocess('mfcc_training_spoof.csv', 1)

X_train = np.vstack((X_bonafide, X_spoof))
y_train = np.hstack((y_bonafide, y_spoof))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)

model = models.Sequential([
    layers.Conv1D(32, 3, activation='relu', input_shape=(13, 1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_reshaped, y_train, epochs=20, batch_size=32, verbose=1)

X_test_bonafide, y_test_bonafide = load_and_preprocess('mfcc_testing_bonafide.csv', 0)
X_test_spoof, y_test_spoof = load_and_preprocess('mfcc_testing_spoof.csv', 1)
X_test = np.vstack((X_test_bonafide, X_test_spoof))
y_test = np.hstack((y_test_bonafide, y_test_spoof))
X_test_scaled = scaler.transform(X_test)
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

test_loss, test_accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)
print(f"Test accuracy: {(test_accuracy*100):.2f}")
print(f"Test loss: {(test_loss):.4f}")

y_pred = model.predict(X_test_reshaped)
y_pred_classes = (y_pred > 0.5).astype(int).flatten()
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=['Bonafide', 'Spoof']))

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_classes, average='binary')
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

eer = compute_eer(y_test, y_pred)
print(f"Equal Error Rate (EER): {(eer*100):.2f}")

p_tar = 0.05  
c_miss_asv = 1
c_fa_asv = 10
c_miss_cm = 1
c_fa_cm = 10

tdcf = compute_tdcf_deepfake(y_test, y_pred, p_tar, c_miss_asv, c_fa_asv, c_miss_cm, c_fa_cm)
print(f"Tandem Detection Cost Function (t-DCF): {tdcf:.4f}")
