import torch
import numpy as np


def sawtooth_wave(x, frequency):
    slope = 4  # cover a distance of -1 to 1 in time frequency/2
    x_scaled = torch.remainder(x * frequency, 0.5)  # cut up x and rescale so it goes from 0 to 1
    # print(x_scaled)
    x_sign = torch.sign(torch.remainder(x * frequency, 1) - x_scaled - 10 ** -10)
    #
    return x_sign - slope * x_scaled * x_sign


class MockNeuralNetwork(torch.nn.Module):
    def __init__(self, num_dim, frequency=1, seed=42):
        super().__init__()
        self.seed = seed
        self.num_dim = num_dim
        self.frequency = frequency

        # Set random seed
        np.random.seed(seed)

        # Random constant projection vector
        # self.projection_vector = tf.constant(np.random.randn(num_dim, 1), dtype=tf.float32)
        # with this calculating robustness is not easy, the below projection is easy
        self.projection_vector = torch.ones((num_dim, 1))

    def raw_output(self, x):
        # Project input onto the random projection vector
        # print(np.shape(inputs))
        # projected_input = tf.matmul(tf.cast(x, dtype=tf.float32), self.projection_vector)
        projected_input = torch.matmul(x, self.projection_vector)
        # print(projected_input)
        # Apply sawtooth function
        return sawtooth_wave(projected_input, self.frequency)

    def forward(self, inputs):
        output = self.raw_output(inputs)
        # Threshold for binary classification
        # output_binary = tf.cast(output > 0.5, dtype=tf.float32)
        output_binary = output > 0.5
        return output_binary


    def robustness(self, threshold):
        """
        returns the fraction of points with a given threshold of robustness.
        Eg for a threshold of 0.5, returns the fraction of points which are more than 0.5 away
        from any point with the opposite sign
        :return: float between 0 and 1 containing the fraction of robust points (considering uniform density)
        """

        # there are 4 regions of size `threshold` per period
        return self.frequency * max(1 / self.frequency - 4 * threshold, 0)


if __name__ == "__main__":
    freq = 1
    dim = 5
    teacher = MockNeuralNetwork(dim, freq)

    # print(teacher.robustness(0.1))
    # Example usage:
    # x = tf.cast(tf.linspace(0.0, 1.0, 1001), dtype=tf.float32)
    x = torch.linspace(0,1,1001)
    x_matrix = x.repeat(dim, 1).t()
    # x_expanded = tf.expand_dims(x, axis=1)
    # x_expanded = torch.unsqueeze(x,1)
    # x_5dim = tf.tile(x_expanded, [1, 5])
    # x_5dim = x
    y = sawtooth_wave(x, 1)

    # print(x_5dim)
    import matplotlib.pyplot as plt

    plt.plot(x, teacher.raw_output(x_matrix))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Zigzag Sawtooth Function')
    plt.grid(True)
    plt.show()

    # ========================
    # now we check the robustness
    # we know the frequency is 0.5, so the slope of the sawtooth is (4*frequency)
    # for a slope of 2, if we want the fraction of points that are a threshold of t away from the decision boundary,
    # we can check for each point, if it's function value is higher |4*freq*t|
    # TODO: does this work with the strange projections as well or am i doing something sketchy?

    t = 0.2
    a = -5
    b = 5
    random_matrix = (b - a) * torch.rand(10**6, dim) + a
    ## the two lines below should return approx the same for all combinations of freq and t
    ## if the analytic method is not applicable, the line below with simulation can be employed regardless :)
    print(torch.mean(
        (
                torch.abs(teacher.raw_output(random_matrix)) > 4 * freq * t
        ).float()))
    print(teacher.robustness(t))
