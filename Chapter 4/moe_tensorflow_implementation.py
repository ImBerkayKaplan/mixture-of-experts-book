import tensorflow as tf
from keras import layers

class Expert(layers.Layer):
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.fc1 = layers.Dense(d_ffn, name='fc1')
        self.activation = layers.Activation(tf.nn.gelu)
        self.dropout = layers.Dropout(dropout)
        self.fc2 = layers.Dense(d_model, name='fc2')

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x