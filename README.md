# AGF_beforeFT_ver1

ECGのみを入力として Arousal・Valence を推定する学習済みモデル。

**モデル概要**
- アーキテクチャ: KD-Ensemble 生徒モデル（α=1.0）
- 学習データ: 2025年飲食実験（100名、30秒窓）
- 入力: ECG HRV特徴量 19次元
- 出力: Arousal, Valence（各スカラー値）

---

## ファイル一覧

| ファイル | 説明 |
|---------|------|
| `model.py` | モデルアーキテクチャ定義（TensorFlow版） |
| `model.pt` | 学習済み重み（PyTorch形式） |
| `checkpoint/` | 学習済み重み（TensorFlow形式） |
| `mean_ecg.npy` | ECG特徴量の正規化用平均値（shape: 19）|
| `std_ecg.npy` | ECG特徴量の正規化用標準偏差（shape: 19）|
| `metadata.json` | モデル設定情報（入力次元・特徴量名など） |
| `feature_extraction.py` | 生ECG波形 → 19次元特徴量への変換スクリプト |
| `inference.py` | 特徴量 → Arousal/Valence 推論サンプルコード（TF版） |
| `convert_to_pt.py` | TFチェックポイント → model.pt 変換スクリプト |

---

## 入力特徴量（19次元）

| # | 特徴量名 | カテゴリ |
|---|---------|---------|
| 1 | NNICounter | 時間領域 |
| 2 | HRMean | 時間領域 |
| 3 | NN20 | 時間領域 |
| 4 | NN30 | 時間領域 |
| 5 | NN50 | 時間領域 |
| 6 | PNN50 | 時間領域 |
| 7 | fftPeakVLF | 周波数領域 |
| 8 | fftPeakLF | 周波数領域 |
| 9 | fftPeakHF | 周波数領域 |
| 10 | fftRelULF | 周波数領域 |
| 11 | fftRelVLF | 周波数領域 |
| 12 | fftRelLF | 周波数領域 |
| 13 | fftRelHF | 周波数領域 |
| 14 | fftLogULF | 周波数領域 |
| 15 | fftLogVLF | 周波数領域 |
| 16 | fftNormLF | 周波数領域 |
| 17 | sampleEntropy | 非線形 |
| 18 | poincareSD2 | 非線形 |
| 19 | poincareSDRatio | 非線形 |

---

## 使い方

### 1. 生ECG波形から特徴量を抽出

```python
# DEEP4環境のみ（ECGFeaturesクラスが必要）
import sys
sys.path.insert(0, '/mnt/learn/usr/hayashi/引継ぎ/プログラム/EmotionRecognition')
from Multimodal.ECG.ECGFeatures import ECGFeatures
from feature_extraction import extract_windows, extract_ecg_features, FS_ECG
import numpy as np

extractor = ECGFeatures(FS_ECG, select_features=True)
ecg_raw = ...  # shape (T,)  生ECG波形 1000Hz

features = []
for window, _ in extract_windows(ecg_raw):
    feat = extract_ecg_features(window, extractor)
    if feat is not None:
        features.append(feat)

X_ecg = np.stack(features)  # (N, 19)
```

### 2. 感情推定（TensorFlow）

```python
from inference import load_model, predict

model, mean_ecg, std_ecg = load_model()
arousal, valence = predict(X_ecg, model, mean_ecg, std_ecg)
```

### 3. 感情推定（PyTorch）

```python
import torch
import numpy as np
from model_pt import RegressionHeadPT  # convert_to_pt.py内のクラスを参照

mean_ecg = np.load('mean_ecg.npy')
std_ecg  = np.load('std_ecg.npy')

model = RegressionHeadPT(input_dim=19, hidden_units=64)
model.load_state_dict(torch.load('model.pt'))
model.eval()

X_norm = (X_ecg - mean_ecg) / std_ecg
x = torch.tensor(X_norm, dtype=torch.float32)
with torch.no_grad():
    arousal, valence = model(x)
```

---

## 処理の流れ

```
生ECG波形（1000Hz）
    ↓ 30秒窓に分割
    ↓ HRV特徴量抽出（ECGFeatures）
19次元特徴量ベクトル
    ↓ mean_ecg / std_ecg で正規化
    ↓ RegressionHead（Dense64 → Arousal, Valence）
Arousal・Valence 予測値
```
