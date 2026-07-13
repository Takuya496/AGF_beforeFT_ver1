"""
Convert TensorFlow checkpoint to PyTorch model.pt
Run this script on DEEP4.

Usage:
    python convert_to_pt.py
"""
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf

CKPT_DIR  = './checkpoint'
MEAN_PATH = './mean_ecg.npy'
STD_PATH  = './std_ecg.npy'
ECG_N     = 19
HIDDEN    = 64


class RegressionHeadPT(nn.Module):
    def __init__(self, input_dim=ECG_N, hidden_units=HIDDEN):
        super().__init__()
        self.hidden  = nn.Linear(input_dim, hidden_units)
        self.out_ar  = nn.Linear(hidden_units, 1)
        self.out_val = nn.Linear(hidden_units, 1)

    def forward(self, x):
        h = torch.nn.functional.elu(self.hidden(x))
        return self.out_ar(h), self.out_val(h)


def load_tf_weights(ckpt_dir):
    """Load TF checkpoint and return weight dict."""
    from model import RegressionHead as TFHead
    tf_model = TFHead(hidden_units=HIDDEN, name='student')
    tf_model(tf.zeros([1, ECG_N]))

    ckpt = tf.train.Checkpoint(student=tf_model)
    ckpt.restore(tf.train.latest_checkpoint(ckpt_dir)).expect_partial()

    weights = {}
    for var in tf_model.trainable_variables:
        weights[var.name] = var.numpy()
        print(f"  {var.name}: {var.shape}")
    return weights


def convert(ckpt_dir=CKPT_DIR, out_path='model.pt'):
    print("Loading TF checkpoint...")
    tf_weights = load_tf_weights(ckpt_dir)

    pt_model = RegressionHeadPT(input_dim=ECG_N, hidden_units=HIDDEN)

    # TF Dense: kernel (in, out), bias (out,)
    # PyTorch Linear: weight (out, in), bias (out,)
    sd = pt_model.state_dict()
    for name, val in tf_weights.items():
        if 'mlp_layer/kernel' in name:
            sd['hidden.weight'] = torch.tensor(val.T, dtype=torch.float32)
        elif 'mlp_layer/bias' in name:
            sd['hidden.bias'] = torch.tensor(val, dtype=torch.float32)
        elif 'out_ar/kernel' in name:
            sd['out_ar.weight'] = torch.tensor(val.T, dtype=torch.float32)
        elif 'out_ar/bias' in name:
            sd['out_ar.bias'] = torch.tensor(val, dtype=torch.float32)
        elif 'out_val/kernel' in name:
            sd['out_val.weight'] = torch.tensor(val.T, dtype=torch.float32)
        elif 'out_val/bias' in name:
            sd['out_val.bias'] = torch.tensor(val, dtype=torch.float32)

    pt_model.load_state_dict(sd)
    pt_model.eval()

    torch.save(pt_model.state_dict(), out_path)
    print(f"Saved: {out_path}")

    # Sanity check
    mean_ecg = np.load(MEAN_PATH).astype(np.float32)
    dummy = torch.zeros(1, ECG_N)
    ar, val = pt_model(dummy)
    print(f"Sanity check — arousal: {ar.item():.4f}, valence: {val.item():.4f}")


if __name__ == '__main__':
    convert()
