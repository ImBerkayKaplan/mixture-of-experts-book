import tensorflow as tf 
import numpy as np 
from tensorflow.keras.losses import KLDivergence 

# ------------------------- Not included in the book -------------------------
import warnings

try:
    from ai_edge_litert import LiteRTInterpreter as Interpreter
    LITE_RT_AVAILABLE = True
except ImportError:
    Interpreter = tf.lite.Interpreter
    LITE_RT_AVAILABLE = False
    warnings.filterwarnings(
        "ignore",
        message=r".*tf\.lite\.Interpreter is deprecated.*",
        category=UserWarning,
        module=r"tensorflow\.lite\.python\.interpreter",
    )

def create_student_model():
    inputs = tf.keras.Input(shape=(32,), name="student_inputs")
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(10, activation="linear", name="student_logits")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="student_model")


def create_teacher_moe_model():
    inputs = tf.keras.Input(shape=(32,), name="teacher_inputs")
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(10, activation="linear", name="teacher_logits")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="teacher_moe_model")


def create_dummy_datasets():
    x_train = np.random.random((64, 32)).astype("float32")
    y_train = np.random.randint(0, 10, size=(64,)).astype("int32")
    x_test = np.random.random((16, 32)).astype("float32")
    y_test = np.random.randint(0, 10, size=(16,)).astype("int32")
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(16)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(16)
    return train_dataset, test_dataset

# ------------------------- Not included in the book -------------------------

class Distiller(tf.keras.Model): 
    def __init__(self, student, teacher): 
        super(Distiller, self).__init__() 
        self.teacher = teacher 
        self.student = student 
    def call(self, inputs, training=False): 
        return self.student(inputs, training=training) 

# ------------------------- Not included in the book -------------------------

    def evaluate(self, x=None, y=None, batch_size=None, verbose="auto", sample_weight=None, steps=None, callbacks=None, return_dict=False, **kwargs):
        if x is None:
            raise ValueError("Evaluation data must be provided.")
        if hasattr(x, "__iter__") and not isinstance(x, (tf.Tensor, np.ndarray)):
            dataset = x
            losses = []
            student_losses = []
            distillation_losses = []
            for batch_x, batch_y in dataset:
                teacher_predictions = self.teacher(batch_x, training=False)
                student_predictions = self.student(batch_x, training=True)
                student_loss = self.student_loss_fn(batch_y, student_predictions)
                distillation_loss = self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                total_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
                losses.append(total_loss.numpy())
                student_losses.append(student_loss.numpy())
                distillation_losses.append(distillation_loss.numpy())
            return [float(np.mean(losses)), float(np.mean(student_losses)), float(np.mean(distillation_losses))]
        raise ValueError("This example expects a dataset iterable for evaluation.")

# ------------------------- Not included in the book -------------------------

    def compile(self, optimizer, metrics, student_loss_fn, distillation_loss_fn, alpha, temperature): 
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics, loss=student_loss_fn) 
        self.student_loss_fn = student_loss_fn 
        self.distillation_loss_fn = distillation_loss_fn 
        self.alpha = alpha 
        self.temperature = temperature 
    def train_step(self, data): 
        x, y = data 
        teacher_predictions = self.teacher(x, training=False) 
        with tf.GradientTape() as tape: 
            student_predictions = self.student(x, training=True) 
            student_loss = self.student_loss_fn(y, student_predictions) 
            distillation_loss = self.distillation_loss_fn(tf.nn.softmax(teacher_predictions /  self.temperature, axis=1), tf.nn.softmax(student_predictions / self.temperature, axis=1)) 
            total_loss = self.alpha * student_loss +  (1 - self.alpha) * distillation_loss 
        trainable_vars = self.student.trainable_variables 
        gradients = tape.gradient(total_loss, trainable_vars) 
        self.optimizer.apply_gradients(zip(gradients, trainable_vars)) 
        self.compiled_metrics.update_state(y, student_predictions) 
        return {m.name: m.result() for m in self.metrics} 

train_dataset, test_dataset = create_dummy_datasets()
teacher_moe_model = create_teacher_moe_model()
student_model = create_student_model()  
distiller = Distiller(student=student_model, teacher=teacher_moe_model) 
distiller.compile( 
    optimizer='adam', 
    metrics=['sparse_categorical_accuracy'], 
    student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    distillation_loss_fn=KLDivergence(), 
    alpha=0.1,      # Give more weight to the distillation loss 
    temperature=10  # Use a high temperature to soften probabilities 
) 

# Train the distiller. This only updates the student model's weights. 
distiller.fit(train_dataset, epochs=10) 

# Evaluate the Student 
results = distiller.evaluate(test_dataset) 
print(f"Total Loss: {results[0]}")  
print(f"Student Loss: {results[1]}")  
print(f"Distillation Loss: {results[2]}") 

# The trained student model is now ready for deployment 
student_model.save("distilled_student_model.keras") 

# Create a small TFLite model if the quantized artifact is missing.
converter = tf.lite.TFLiteConverter.from_keras_model(student_model)
quantized_model = converter.convert()
with open("quantized_moe_student.tflite", "wb") as f:
    f.write(quantized_model)

interpreter = Interpreter(model_path="quantized_moe_student.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape'] 
input_dtype = input_details[0]['dtype'] 
input_data = np.array(np.random.random_sample(input_shape), dtype=input_dtype) 

interpreter.set_tensor(input_details[0]['index'], input_data) 
interpreter.invoke() 
output_data = interpreter.get_tensor(output_details[0]['index']) 