import numpy as np
# import tensorflow as tf
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from Dataset import SyntheticDataset
from RobustMockTeacher import MockNeuralNetwork


# loss function (same as the one in Definition 1.3 (form Shao et al) in the overleaf document)
# y_true is the true label (from training dataset)
# y_pred_S and y_pred_T are the predictions of the student and teacher models respectively
# def LCE(y_true, y_pred):
#     return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
#
#
# def LKL(y_pred_S, y_pred_T, temperature=1.0):
#     y_pred_S /= temperature
#     y_pred_T /= temperature
#     return tf.keras.losses.KLDivergence(y_pred_S, y_pred_T)
#
#
# def LGAD_tf(x, y_true, y_pred_S, y_pred_T, lambda_CE, lambda_KL, lambda_GAD, temperature=1.0):
#     # Cross-entropy loss
#     CE_loss = LCE(y_true, y_pred_S)
#
#     # KL-divergence loss
#     KL_loss = temperature ** 2 * LKL(y_pred_S / temperature, y_pred_T / temperature)
#
#     # Compute gradients of CE loss w.r.t. input x for both student and teacher models
#     with tf.GradientTape() as tape_S:
#         tape_S.watch(x)
#         CE_loss_S = LCE(y_true, y_pred_S)
#         grad_CE_S = tape_S.gradient(CE_loss_S, x)
#
#     with tf.GradientTape() as tape_T:
#         tape_T.watch(x)
#         CE_loss_T = LCE(y_true, y_pred_T)
#         grad_CE_T = tape_T.gradient(CE_loss_T, x)
#
#     # Gradient discrepancy loss
#     grad_discrepancy = tf.norm(grad_CE_S - grad_CE_T)
#
#     # Combine losses
#     LGAD_loss = lambda_CE * CE_loss + lambda_KL * KL_loss + lambda_GAD * grad_discrepancy
#
#     return LGAD_loss


def LGAD(x, y_true, y_pred_S, y_pred_T, lambda_CE=1.0, lambda_KL=1.0, lambda_GAD=1.0,
         temperature=1.0):
    LCE = torch.nn.CrossEntropyLoss()
    LKL = torch.nn.KLDivLoss(log_target=True)
    # print(y_true)
    # print(y_pred_T)
    CE_loss = LCE(y_pred_S, y_true)
    x.requires_grad = True
    KL_loss = temperature ** 2 * LKL(torch.nn.functional.log_softmax(y_pred_S, dim=1),
                                     torch.nn.functional.log_softmax(y_pred_T, dim=1))

    CE_loss_T = -LCE(y_pred_T, y_true)
    # print(y_pred_T)
    CE_loss_T.backward(retain_graph=True)
    # optimizer.zero_grad()
    CE_loss.backward(retain_graph=True)
    grad_discrepancy = torch.norm(x.grad)
    # print(grad_discrepancy)
    # PLEASE HELP: is this equivalent to the above code?
    # i think gradients accumulate here naturally
    # print(CE_loss_T)
    print(f"CE:{CE_loss},KL:{KL_loss},GAD:{grad_discrepancy}")

    return lambda_CE * CE_loss + lambda_KL * KL_loss + lambda_GAD * grad_discrepancy


def knowledge_distillation(teacher_model, student_model, num_samples, input_shape, batch_size, epochs):
    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    # Compile teacher model
    # teacher_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Generate synthetic data using normal distribution
    synthetic_data = torch.tensor(np.random.normal(size=(num_samples, *input_shape)),dtype=torch.float32)

    # Get teacher predictions for synthetic data
    teacher_predictions = teacher_model(synthetic_data)

    # Convert teacher predictions to labels
    synthetic_labels = torch.eye(2)[torch.argmax(teacher_predictions, dim=1)]#.reshape(-1, 1)
    # print(synthetic_labels)
    # synthetic_labels = tf.keras.utils.to_categorical(synthetic_labels)

    # Train student model using synthetic data and teacher predictions
    # student_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    optimizer = optim.Adam(student_model.parameters(), lr=0.0005)

    dataset = SyntheticDataset(synthetic_data, synthetic_labels)

    # Create DataLoader
    for epoch in tqdm(range(100)):
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


        for inputs, targets in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            inputs.requires_grad = True

            # Forward pass
            outputs = student_model(inputs)
            teacher_outputs = teacher_model(inputs)
            # print(teacher_outputs)
            # Set requires_grad=True on inputs to enable gradient computation

            # Compute your custom loss
            loss = LGAD(inputs, targets, outputs, teacher_outputs, temperature=np.exp(-epoch/20))
            history["loss"].append(loss.detach().numpy())
            # history["accuracy"].append().numpy()
            # print(loss.detach().numpy())
            loss.backward()
            # print(inputs.grad)
            # Optimize (update weights)
            optimizer.step()
    return history
    #
    # student_model.compile(optimizer='adam',
    #                       loss=lambda y_true, y_pred: LGAD(inputs, y_true, y_pred, teacher_model(inputs),
    #                                                        lambda_CE, lambda_KL, lambda_GAD, temperature),
    #                       metrics=['accuracy'])
    # student_model.fit(synthetic_data, synthetic_labels, batch_size=batch_size, epochs=epochs)


if __name__ == "__main__":
    teacher = MockNeuralNetwork(5, 1)
    knowledge_distillation(teacher, teacher, 10 ** 6, (5,), 1000, 100)
