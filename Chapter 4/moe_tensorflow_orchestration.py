import tensorflow as tf
from keras import layers
import einops # Using einops for clarity in tensor manipulation

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

class MoELayer(layers.Layer):
    def __init__(self,
                 d_model: int,
                 num_experts: int,
                 top_k: int,
                 d_ffn: int,
                 dropout: float = 0.1,
                 aux_loss_alpha: float = 0.01,
                 **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_alpha = aux_loss_alpha

        self.gate = layers.Dense(self.num_experts, use_bias=False, name='gate')

        self. experts = [Expert(d_model, d_ffn, dropout, name=f'expert_{i}') for i in range(self.num_experts)]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        original_shape = tf.shape(x)
        batch_size = original_shape[0]
        seq_len = original_shape[1]

        x_reshaped = tf.reshape(x, [-1, self.d_model])
        num_tokens = tf.shape(x_reshaped)[0]

        gate_logits = self.gate(x_reshaped)

        routing_weights, selected_experts = tf.math.top_k(gate_logits, k=self.top_k)
        routing_weights = tf.nn.softmax(routing_weights, axis=-1)

        gate_probs = tf.nn.softmax(tf.cast(gate_logits, tf.float32), axis=-1)
        
        expert_mask = tf.one_hot(selected_experts, self.num_experts)
        tokens_per_expert = tf.reduce_sum(tf.reduce_sum(expert_mask, axis=1), axis=0)
        fraction_tokens_per_expert = tokens_per_expert / tf.cast(num_tokens, tf.float32)

        probs_per_expert = tf.reduce_mean(gate_probs, axis=0)

        aux_loss = self.aux_loss_alpha * float(self.num_experts) * tf.tensordot(fraction_tokens_per_expert, probs_per_expert, axes=1)
        self.add_loss(aux_loss)

        token_indices = tf.range(num_tokens, dtype=tf.int32)
        token_indices = einops.repeat(token_indices, 't -> t k', k=self.top_k)
        dispatch_indices = tf.stack([tf.reshape(token_indices, [-1]), tf.reshape(selected_experts, [-1])], axis=1)

        dispatch_tensor = tf.SparseTensor(
            indices=tf.cast(dispatch_indices, tf.int64),
            values=tf.reshape(routing_weights, [-1]),
            dense_shape=[num_tokens, self.num_experts]
        )
        dispatch_tensor = tf.sparse.reorder(dispatch_tensor)

        final_output = tf.zeros_like(x_reshaped)
        
        for i in range(self.num_experts):
            expert_weights = tf.sparse.slice(dispatch_tensor, start=[0, i], size=[num_tokens, 1])
            expert_weights = tf.sparse.reshape(expert_weights, shape=[-1])
            expert_weights = tf.sparse.to_dense(expert_weights)

            active_token_indices = tf.where(expert_weights > 0)
            active_token_indices = tf.squeeze(active_token_indices, axis=1)

            if tf.size(active_token_indices) > 0:
                expert_inputs = tf.gather(x_reshaped, active_token_indices)
                
                expert_output = self.experts[i](expert_inputs)
                
                active_routing_weights = tf.gather(expert_weights, active_token_indices)
                active_routing_weights = tf.expand_dims(active_routing_weights, axis=-1)
                
                weighted_output = expert_output * active_routing_weights
                
                final_output = tf.tensor_scatter_nd_add(
                    final_output,
                    tf.expand_dims(active_token_indices, axis=1),
                    weighted_output
                )

        final_output = tf.reshape(final_output, original_shape)
        
        return final_output