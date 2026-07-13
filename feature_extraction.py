"""
ECG feature extraction for AGF_beforeFT_ver1

Extracts 19 HRV features from a 30-second ECG signal.
These features are the input to the student model (RegressionHead).

Requirements
------------
- ECGFeatures class from EmotionRecognition/Multimodal/ECG/ECGFeatures.py
  (located in the EmotionRecognition repository on DEEP4)

Sampling frequency : 1000 Hz  (FS_ECG_EDE)
Window length      : 30 s     (SPLIT_TIME)
Stride             : 30 s     (no overlap, STRIDE=1.0)
Input shape        : (30000,) raw ECG signal
Output shape       : (19,)    feature vector
"""

import sys
import numpy as np

# ---- ECGFeatures dependency (DEEP4 only) ----
# sys.path.insert(0, '/mnt/learn/usr/hayashi/引継ぎ/プログラム/EmotionRecognition')
# from Multimodal.ECG.ECGFeatures import ECGFeatures

FS_ECG    = 1000    # Hz
WIN_SEC   = 30      # seconds
WIN_SAMP  = FS_ECG * WIN_SEC          # 30000 samples
MIN_SAMP  = (FS_ECG - 2) * WIN_SEC   # 29940 samples (tolerance)
ECG_N     = 19

ECG_SELECTED_LIST = [
    "NNICounter", "HRMean", "NN20", "NN30", "NN50", "PNN50",
    "fftPeakVLF", "fftPeakLF", "fftPeakHF",
    "fftRelULF", "fftRelVLF", "fftRelLF", "fftRelHF",
    "fftLogULF", "fftLogVLF", "fftNormLF",
    "sampleEntropy", "poincareSD2", "poincareSDRatio",
]


def extract_ecg_features(ecg_signal, ecg_features_extractor):
    """
    Extract 19 HRV features from a single 30-second ECG window.

    Parameters
    ----------
    ecg_signal : np.ndarray  shape (WIN_SAMP,)  raw ECG at 1000 Hz
    ecg_features_extractor : ECGFeatures instance
        Created as: ECGFeatures(FS_ECG, select_features=True)

    Returns
    -------
    features : np.ndarray  shape (19,)  or None if extraction failed
    """
    ecg = ecg_signal[:MIN_SAMP]
    if ecg.shape[0] != MIN_SAMP:
        return None

    time_domain    = ecg_features_extractor.extractTimeDomain_(ecg)
    freq_domain    = ecg_features_extractor.extractFrequencyDomain_(ecg)
    nonlinear      = ecg_features_extractor.extractNonLinearDomain_(ecg)

    if time_domain.shape[0] == 0 or freq_domain.shape[0] == 0 or nonlinear.shape[0] == 0:
        return None

    features = np.concatenate([time_domain, freq_domain, nonlinear])
    if np.any(np.isinf(features)) or np.any(np.isnan(features)):
        return None

    assert features.shape[0] == ECG_N, f"Expected {ECG_N} features, got {features.shape[0]}"
    return features


def extract_windows(ecg_signal, stride_sec=WIN_SEC):
    """
    Slide a 30-second window over a long ECG signal.

    Parameters
    ----------
    ecg_signal  : np.ndarray  (T,)   raw ECG at 1000 Hz
    stride_sec  : float              stride in seconds (default=30, no overlap)

    Yields
    ------
    window : np.ndarray  (WIN_SAMP,)
    start  : int         start sample index
    """
    stride = int(FS_ECG * stride_sec)
    for start in range(0, len(ecg_signal) - WIN_SAMP, stride):
        yield ecg_signal[start:start + WIN_SAMP], start


# ---------------------------------------------------------------------------
# Usage example (DEEP4 environment)
# ---------------------------------------------------------------------------
# sys.path.insert(0, '/mnt/learn/usr/hayashi/引継ぎ/プログラム/EmotionRecognition')
# from Multimodal.ECG.ECGFeatures import ECGFeatures
#
# extractor = ECGFeatures(FS_ECG, select_features=True)
#
# ecg_raw = pd.read_csv('filtered_XXX_ECG.csv')['ECG'].values   # shape (T,)
# feature_list = []
# for window, start in extract_windows(ecg_raw):
#     feat = extract_ecg_features(window, extractor)
#     if feat is not None:
#         feature_list.append(feat)
# X_ecg = np.stack(feature_list)   # (N, 19) → input to inference.py
