"""
ECG-only emotion estimation model (student, KD-Ensemble α=1.0)

Input : ECG feature vector (N, ECG_N)  -- normalized
Output: (arousal, valence) each (N, 1)
"""
import tensorflow as tf


class RegressionHead(tf.keras.Model):
    def __init__(self, hidden_units=64, **kwargs):
        super().__init__(**kwargs)
        self.hidden  = tf.keras.layers.Dense(hidden_units, activation='elu', name='mlp_layer')
        self.out_ar  = tf.keras.layers.Dense(1, name='out_ar')
        self.out_val = tf.keras.layers.Dense(1, name='out_val')

    def call(self, z, training=None):
        h = self.hidden(z)
        return self.out_ar(h), self.out_val(h)
