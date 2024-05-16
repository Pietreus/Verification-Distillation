import numpy as np
import torch


from RobustMockTeacher import MockNeuralNetwork

#loss function (same as the one in Definition 1.3 (form Shao et al) in the overleaf document)
#y_true is the true label (from training dataset) 
#y_pred_S and y_pred_T are the predictions of the student and teacher models respectively


def LCE(y_true, y_pred):
    ce_loss = torch.keras.losses.BinaryCrossentropy(from_logits=False)
    return ce_loss(y_true, y_pred)


def LKL(y_pred_S, y_pred_T, temperature=1.0):
    y_pred_S /= temperature
    y_pred_T /= temperature
    kl_loss = torch.keras.losses.KLDivergence()
    return kl_loss(y_pred_S, y_pred_T)


def LGAD(x, y_true, y_pred_S, y_pred_T, gradient_tape, lambda_CE, lambda_KL, lambda_GAD, temperature=1.0):

    y_true = tf.cast(y_true,dtype=tf.float32)
    # Cross-entropy loss
    CE_loss = LCE(y_true, y_pred_S)
    # KL-divergence loss
    KL_loss = temperature**2 * LKL(y_pred_S / temperature, y_pred_T / temperature)
    
    # Compute gradients of CE loss w.r.t. input x for both student and teacher models
    CE_loss_S = LCE(y_true, y_pred_S)
    grad_CE_S = gradient_tape.gradient(CE_loss_S, x)

    # CE_loss_T = LCE(y_true, y_pred_T)
    # grad_CE_T = gradient_tape.gradient(CE_loss_T, x)

    print(grad_CE_S)
    # Gradient discrepancy loss
    grad_discrepancy = tf.norm(grad_CE_S - grad_CE_T)
    
    # Combine losses
    LGAD_loss = lambda_CE * CE_loss + lambda_KL * KL_loss + lambda_GAD * grad_discrepancy
    
    return LGAD_loss



def knowledge_distillation(teacher_model, student_model, num_samples, input_shape, batch_size, epochs,
                           lambda_CE=1, lambda_KL=1, lambda_GAD=1, temperature=1):
    # Compile teacher model
    teacher_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Generate synthetic data using normal distribution
    synthetic_data = np.random.normal(size=(num_samples, *input_shape))
    
    # Get teacher predictions for synthetic data
    teacher_predictions = teacher_model.predict(synthetic_data)
    
    # Convert teacher predictions to labels
    synthetic_labels = np.argmax(teacher_predictions, axis=1)
    synthetic_labels = tf.keras.utils.to_categorical(synthetic_labels)
    
    # Train student model using synthetic data and teacher predictions
    #student_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    print("Student training starting")

   
    def lgad_train_step(inputs, y_true):
        with tf.GradientTape() as tape:
            # Forward pass
            tape.watch(inputs)
            y_pred = student_model(inputs, training=True)
            teacher_pred = teacher_model(inputs, training=False)
            # Compute the loss

            loss = LGAD(inputs, y_true, y_pred, teacher_pred, tape, lambda_CE, lambda_KL, lambda_GAD, temperature)

        # Compute gradients
        gradients = tape.gradient(loss, student_model.trainable_variables)

        # Update weights
        optimizer.apply_gradients(zip(gradients, student_model.trainable_variables))

        # Update metrics
        train_loss(loss)
        train_accuracy(y_true, y_pred)

    # student_model.compile(optimizer='adam', loss=lambda input, y_true, y_pred: LGAD(input, y_true, y_pred, teacher_model(input),
    #     lambda_CE, lambda_KL, lambda_GAD, temperature), metrics=['accuracy'],)

    EPOCHS = 10
    train_dataset = tf.data.Dataset.from_tensor_slices((synthetic_data, synthetic_labels))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32)
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_state()
        train_accuracy.reset_state()

        for inputs, y_true in train_dataset:
            lgad_train_step(inputs, y_true)

    return student_model.fit(synthetic_data, synthetic_labels, batch_size=batch_size, epochs=epochs, verbose=1)



if __name__ == "__main__":
    teacher = MockNeuralNetwork(42, 5, 1)
    knowledge_distillation(teacher,teacher,10**6,(5, ),1000,100)

