"""
Inference example for AGF_beforeFT_ver1

Trained on: 2025 eating experiment (100 subjects, 30s windows, α=1.0 KD)
Input : ECG feature vector  shape (N, ECG_N)  -- raw (not normalized)
Output: arousal, valence     each shape (N,)
"""
import numpy as np
import tensorflow as tf
from model import RegressionHead

CKPT_DIR = './checkpoint'
MEAN_ECG = './mean_ecg.npy'
STD_ECG  = './std_ecg.npy'


def load_model(ckpt_dir=CKPT_DIR, mean_path=MEAN_ECG, std_path=STD_ECG):
    mean_ecg = np.load(mean_path)
    std_ecg  = np.load(std_path)

    model = RegressionHead(hidden_units=64, name='student')
    # build with dummy input
    dummy = tf.zeros([1, mean_ecg.shape[0]])
    model(dummy)

    ckpt = tf.train.Checkpoint(student=model)
    status = ckpt.restore(tf.train.latest_checkpoint(ckpt_dir))
    status.expect_partial()
    model.trainable = False

    return model, mean_ecg, std_ecg


def predict(X_ecg, model=None, mean_ecg=None, std_ecg=None):
    """
    Parameters
    ----------
    X_ecg : np.ndarray  shape (N, ECG_N)  raw ECG features (30s window)

    Returns
    -------
    arousal  : np.ndarray  (N,)
    valence  : np.ndarray  (N,)
    """
    if model is None:
        model, mean_ecg, std_ecg = load_model()

    X = (np.asarray(X_ecg, dtype=np.float32) - mean_ecg) / std_ecg
    ar_out, val_out = model(X, training=False)
    return ar_out.numpy().flatten(), val_out.numpy().flatten()


if __name__ == '__main__':
    model, mean_ecg, std_ecg = load_model()
    print('Model loaded.')

    # example: replace with your ECG features
    # X_ecg = np.load('your_ecg_features.npy')  # (N, ECG_N)
    # arousal, valence = predict(X_ecg, model, mean_ecg, std_ecg)
    # print(f'arousal : {arousal}')
    # print(f'valence : {valence}')
