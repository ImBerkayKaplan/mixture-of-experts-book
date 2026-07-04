# ------------------------- Not included in the book -------------------------

import tensorflow as tf
from tensorflow import keras
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.quantization.keras import quantize_config
from tensorflow_model_optimization.python.core.quantization.keras import quantizers

@keras.utils.register_keras_serializable()
class MoEQuantizeConfig(quantize_config.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return [(layer.gate.kernel, quantizers.LastValueQuantizer(num_bits=8, per_axis=False, symmetric=True, narrow_range=False))]

    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_weights(self, layer, quantize_weights):
        layer.gate.kernel = quantize_weights[0]

    def set_quantize_activations(self, layer, quantize_activations):
        pass

    def get_output_quantizers(self, layer):
        return []

    def get_config(self):
        return {}


@keras.utils.register_keras_serializable()
class Expert(keras.layers.Layer):
    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.fc1 = keras.layers.Dense(d_ffn, name="fc1")
        self.activation = keras.layers.Activation(tf.nn.gelu)
        self.dropout_layer = keras.layers.Dropout(dropout)
        self.fc2 = keras.layers.Dense(d_model, name="fc2")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout_layer(x)
        x = self.fc2(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "d_ffn": self.d_ffn,
            "dropout": self.dropout,
        })
        return config


@keras.utils.register_keras_serializable()
class MoELayer(keras.layers.Layer):
    def __init__(self, d_model: int, num_experts: int, top_k: int, d_ffn: int, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(int(top_k), int(num_experts))
        self.d_ffn = d_ffn
        self.dropout = dropout
        self.gate = keras.layers.Dense(self.num_experts, use_bias=False, name="gate")
        self.experts = [
            Expert(d_model, d_ffn, dropout, name=f"expert_{idx}")
            for idx in range(self.num_experts)
        ]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        routing_logits = self.gate(x)
        routing_weights = tf.nn.softmax(routing_logits, axis=-1)
        top_k_values, top_k_indices = tf.math.top_k(routing_weights, k=self.top_k)
        top_k_values = tf.nn.softmax(top_k_values, axis=-1)

        expert_outputs = tf.stack([expert(x) for expert in self.experts], axis=1)
        selected_expert_outputs = tf.gather(expert_outputs, top_k_indices, batch_dims=1, axis=1)
        weighted_outputs = selected_expert_outputs * tf.expand_dims(top_k_values, axis=-1)
        return tf.reduce_sum(weighted_outputs, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "d_ffn": self.d_ffn,
            "dropout": self.dropout,
        })
        return config


def create_moe_model(input_dim: int = 32, num_classes: int = 10, num_experts: int = 2, d_model: int = 32, d_ffn: int = 64, top_k: int = 1, dropout: float = 0.1):
    inputs = keras.Input(shape=(input_dim,), name="inputs")
    x = keras.layers.Dense(d_model, activation="relu", name="input_projection")(inputs)
    moe_layer = MoELayer(d_model=d_model, num_experts=num_experts, top_k=top_k, d_ffn=d_ffn, dropout=dropout, name="moe_layer")
    x = tfmot.quantization.keras.quantize_annotate_layer(moe_layer, quantize_config=MoEQuantizeConfig())(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    return keras.Model(inputs=inputs, outputs=outputs, name="moe_classifier")


def make_train_dataset(batch_size: int = 32, input_dim: int = 32, num_classes: int = 10, num_samples: int = 256):
    features = tf.random.normal((num_samples, input_dim))
    labels = tf.random.uniform((num_samples,), minval=0, maxval=num_classes, dtype=tf.int32)
    return tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ------------------------- Not included in the book -------------------------

with keras.utils.custom_object_scope({"MoEQuantizeConfig": MoEQuantizeConfig, "Expert": Expert, "MoELayer": MoELayer}):
    base_model = create_moe_model()
    quantize_aware_model = tfmot.quantization.keras.quantize_apply(base_model)

quantize_aware_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

train_dataset = make_train_dataset()
quantize_aware_model.fit(train_dataset, epochs=1)

converter = tf.lite.TFLiteConverter.from_keras_model(quantize_aware_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert() 