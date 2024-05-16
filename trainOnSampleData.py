import tensorflow as tf
import matplotlib.pyplot as plt

import KD


def create_teacher(X_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(3, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_student():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4, activation='relu', input_shape=(X_train.shape[1],)),#slightly smaller
        tf.keras.layers.Dense(2, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model




def train_and_plot_progress(history):

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate binary classification data
    X, y = make_classification(n_samples=10 ** 4, n_features=5, n_classes=2, random_state=42)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the model
    teacher = create_teacher(X_train)

    # Train the model and plot the progress
    print("Teacher training")
    history = teacher.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose=1)
    train_and_plot_progress(history)

    student = create_student()
    history = KD.knowledge_distillation(teacher,student,10**4, (5,),100,100)
    train_and_plot_progress(history)


