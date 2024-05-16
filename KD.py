import keras
import numpy as np
import tensorflow as tf

from Distiller import Distiller
from RobustMockTeacher import MockNeuralNetwork


# Loss function definitions
def LCE(y_true, y_pred):
    ce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    return ce_loss(y_true, y_pred)


def LKL(y_pred_S, y_pred_T, temperature=1.0):
    y_pred_S /= temperature
    y_pred_T /= temperature
    kl_loss = tf.keras.losses.KLDivergence()
    return kl_loss(y_pred_T, y_pred_S)


def knowledge_distillation(teacher_model, student_model, num_samples, input_shape, batch_size, epochs,
                           lambda_CE=1, lambda_KL=1, lambda_GAD=1, temperature=1):
    def LGAD(x, y_true, y_pred_S, y_pred_T, lambda_CE, lambda_KL, lambda_GAD, temperature=1.0):
        y_true = tf.cast(y_true, dtype=tf.float32)

        # Cross-entropy loss
        CE_loss = LCE(y_true, y_pred_S)

        # KL-divergence loss
        KL_loss = temperature ** 2 * LKL(y_pred_S, y_pred_T, temperature)

        # Compute gradients of CE loss w.r.t. input x for the student model
        with tf.GradientTape() as tape_S:
            tape_S.watch(x)
            y_pred_S_for_grad = student_model(x, training=True)
            CE_loss_S = LCE(y_true, y_pred_S_for_grad)
        grad_CE_S = tape_S.gradient(CE_loss_S, x)

        # Compute gradients of CE loss w.r.t. input x for the teacher model
        with tf.GradientTape() as tape_T:
            tape_T.watch(x)
            y_pred_T_for_grad = teacher_model(x, training=False)
            CE_loss_T = LCE(y_true, y_pred_T_for_grad)
        grad_CE_T = tape_T.gradient(CE_loss_T, x)

        grad_discrepancy = tf.norm(grad_CE_S - grad_CE_T)

        # Combine losses
        LGAD_loss = lambda_CE * CE_loss + lambda_KL * KL_loss + lambda_GAD * grad_discrepancy

        return LGAD_loss

    # Compile teacher model
    teacher_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Generate synthetic data using normal distribution
    synthetic_data = np.random.normal(size=(num_samples, *input_shape))

    # Get teacher predictions for synthetic data
    teacher_predictions = teacher_model.predict(synthetic_data)

    # Convert teacher predictions to labels
    synthetic_labels = np.argmax(teacher_predictions, axis=1)
    synthetic_labels = tf.keras.utils.to_categorical(synthetic_labels)

    # optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_loss = keras.metrics.SparseCategoricalAccuracy()
    train_loss = keras.metrics.BinaryAccuracy()
    # train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
    # Initialize and compile distiller
    distiller = Distiller(student=student_model, teacher=teacher_model)
    distiller.compile(
        optimizer=keras.optimizers.Adam(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.1,
        temperature=10,
    )

    # Distill teacher to student
    distiller.fit(synthetic_data, synthetic_labels, epochs=3, verbose=1,)

    return student_model


if __name__ == "__main__":
    teacher = MockNeuralNetwork(42, 5, 1)
    student = MockNeuralNetwork(42, 5, 1)  # Assuming the student has the same architecture for simplicity
    trained_student = knowledge_distillation(teacher, student, 10 ** 6, (5,), 1000, 10)