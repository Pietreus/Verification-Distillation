import tensorflow as tf
import numpy as np


def sawtooth_wave(x, frequency):
    slope = 4 # cover a distance of -1 to 1 in time frequency/2
    x_scaled = tf.math.mod(x*frequency, 0.5) # cut up x and rescale so it goes from 0 to 1
    # print(x_scaled)
    x_sign = tf.math.sign(tf.math.mod(x*frequency, 1) - x_scaled - 10**-10)
    #
    return (x_sign - slope*x_scaled * x_sign)


class MockNeuralNetwork(tf.keras.Model):
    def __init__(self, seed, num_dim, frequency):
        super(MockNeuralNetwork, self).__init__()
        self.seed = seed
        self.num_dim = num_dim
        self.frequency = frequency

        # Set random seed
        np.random.seed(seed)

        # Random constant projection vector
        #self.projection_vector = tf.constant(np.random.randn(num_dim, 1), dtype=tf.float32)
        # with this calculating robustness is not easy, the below projection is easy
        self.projection_vector = tf.constant(np.ones((num_dim, 1)), dtype=tf.float32)

    def forward(self, inputs):
        # Project input onto the random projection vector
        # print(np.shape(inputs))
        projected_input = tf.matmul(tf.cast(inputs, dtype=tf.float32), self.projection_vector)
        # print(projected_input)
        # Apply sawtooth function
        return sawtooth_wave(projected_input, self.frequency)

    def call(self, inputs):
        output = self.forward(inputs)
        # Threshold for binary classification
        output_binary = tf.cast(output > 0.5, dtype=tf.float32)
        return output_binary

    def compile(self, optimizer, loss, metrics=None):
        pass

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0,
            validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0,
            steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1,
            max_queue_size=10, workers=1, use_multiprocessing=False):
        pass


    def predict(self, x, batch_size=None, verbose=0, steps=None, callbacks=None, max_queue_size=10, workers=1,
                use_multiprocessing=False):
        # Dummy predict method, just return zeros or random values
        return self.call(x)

    def robustness(self, threshold):
        """
        returns the fraction of points with a given threshold of robustness.
        Eg for a threshold of 0.5, returns the fraction of points which are more than 0.5 away
        from any point with the opposite sign
        :return: float between 0 and 1 containing the fraction of robust points (considering uniform density)
        """

        # there are 4 regions of size `threshold` per period
        return self.frequency*tf.math.maximum(1/self.frequency - 4*threshold, 0)


if __name__ == "__main__":

    freq = 1
    teacher = MockNeuralNetwork(42, 5, freq)

    # print(teacher.robustness(0.1))
    # Example usage:
    x = tf.cast(tf.linspace(0.0, 1.0, 1001),dtype=tf.float32)
    x_expanded = tf.expand_dims(x, axis=1)
    x_5dim = tf.tile(x_expanded, [1, 5])
    y = sawtooth_wave(x, 1)

    # print(x_5dim)
    import matplotlib.pyplot as plt
    plt.plot(x, teacher.forward(x_5dim))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Zigzag Sawtooth Function')
    plt.grid(True)
    plt.show()

    #========================
    # now we check the robustness
    # we know the frequency is 0.5, so the slope of the sawtooth is (4*frequency)
    # for a slope of 2, if we want the fraction of points that are a threshold of t away from the decision boundary,
    # we can check for each point, if it's function value is higher |4*freq*t|
    #TODO: does this work with the strange projections as well or am i doing something sketchy?

    t = 0.2

    random_matrix = tf.random.uniform(shape=(10**6, 5), minval=-5, maxval=5,dtype=tf.float32)
    ## the two lines below should return approx the same for all combinations of freq and t
    ## if the analytic method is not applicable, the line below with simulation can be employed regardless :)
    print(tf.math.reduce_mean(tf.cast(tf.math.abs(teacher.forward(random_matrix)) > 4*freq*t, dtype=tf.float32)))
    print(teacher.robustness(t))
