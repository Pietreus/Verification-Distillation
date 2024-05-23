import torch
import numpy as np
from torch.autograd import Function


class Sawtooth_wave(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, frequency = 1):
        slope = 4  # cover a distance of -1 to 1 in time frequency/2
        x_scaled = torch.remainder(input * frequency, 0.5)  # cut up x and rescale so it goes from 0 to 1
        # print(x_scaled)
        x_sign = torch.sign(torch.remainder(input * frequency, 1) - x_scaled - 10 ** -10)
        # x_sign[torch.abs(input)<0.2] = -x_sign[torch.abs(input)<0.2]
        ctx.save_for_backward(x_sign)

        #
        return x_sign - slope * x_scaled * x_sign
        # return x_sign*slope/4 - slope * x_scaled * x_sign

    @staticmethod
    def backward(ctx, grad_output):
        x_sign, = ctx.saved_tensors
        grad_input = -4 * x_sign * grad_output
        return grad_input, None

# def sawtooth_wave(x, frequency):
#     slope = 4  # cover a distance of -1 to 1 in time frequency/2
#     x_scaled = torch.remainder(x * frequency, 0.5)  # cut up x and rescale so it goes from 0 to 1
#     print(x_scaled)
    # x_sign = torch.sign(torch.remainder(x * frequency, 1) - x_scaled - 10 ** -10)

    # return x_sign - slope * x_scaled * x_sign

class Squaretooth_wave(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, frequency = 1):
        slope = 4  # cover a distance of -1 to 1 in time frequency/2
        x_scaled = torch.remainder(input * frequency, 0.5)  # cut up x and rescale so it goes from 0 to 1
        # print(x_scaled)
        x_sign = torch.sign(torch.remainder(input * frequency, 1) - x_scaled - 10 ** -10)
        # x_sign[torch.abs(input)<0.2] = -x_sign[torch.abs(input)<0.2]
        ctx.save_for_backward(x_sign)

        return x_sign
        # return x_sign*slope/4 - slope * x_scaled * x_sign

    @staticmethod
    def backward(ctx, grad_output):
        x_sign, = ctx.saved_tensors
        return torch.zeros_like(x_sign), None

# def sawtooth_wave(x, frequency):
#     slope = 4  # cover a distance of -1 to 1 in time frequency/2
#     x_scaled = torch.remainder(x * frequency, 0.5)  # cut up x and rescale so it goes from 0 to 1
#     print(x_scaled)
# x_sign = torch.sign(torch.remainder(x * frequency, 1) - x_scaled - 10 ** -10)

# return x_sign - slope * x_scaled * x_sign




class MockNeuralNetwork(torch.nn.Module):
    def __init__(self, num_dim, frequency=1, seed=42, device="cpu"):
        super().__init__()
        self.seed = seed
        self.num_dim = num_dim
        self.frequency = frequency
        self.sawtooth = Sawtooth_wave()

        # Set random seed
        np.random.seed(seed)

        # Random constant projection vector
        # with this calculating robustness is not easy, the below projection is easy
        self.projection_vector = torch.ones((num_dim, 1), dtype=torch.float, requires_grad=True).to(device)

    def raw_output(self, x):
        # Project input onto the random projection vector
        projected_input = torch.matmul(x, self.projection_vector)
        # Apply sawtooth function
        return Squaretooth_wave().apply(projected_input, self.frequency)
        # return Squaretooth_wave().apply(projected_input, self.frequency)

    def forward(self, inputs):

        output = self.raw_output(inputs)

        # # Threshold for binary classification
        # output_binary = (output > 0.5)
        # # print(output_binary.view(-1).long())
        # output_two_call = torch.eye(2)[output_binary.view(-1).long()]
        # return output_two_call
        return torch.cat((output, -output), dim=1)

    def backward(self, grad_outputs):
        return Squaretooth_wave.backward(grad_outputs)
        # return Squaretooth_wave.backward(grad_outputs)

    def theoretical_robustness(self, threshold):
        """
        returns the fraction of points with a given threshold of robustness.
        Eg for a threshold of 0.5, returns the fraction of points which are more than 0.5 away
        from any point with the opposite sign
        :return: float between 0 and 1 containing the fraction of robust points (considering uniform density)
        """

        # there are 4 regions of size `threshold` per period
        return self.frequency * max(1 / self.frequency - 4 * threshold, 0)

    def data_robustness(self, data_matrix: torch.Tensor, threshold: float):
        """

        :param data_matrix: the data against which the robustness should be tested
        :param threshold: which robustness radius should be checked
        :return: a fraction of points which are threshold-robust
        """
        return torch.mean((torch.abs(self.raw_output(data_matrix)) > 4 * self.frequency * threshold).float())

    def parameters(self, recurse: bool = True):
        return []


if __name__ == "__main__":
    freq = 1
    dim = 5
    teacher = MockNeuralNetwork(dim, freq)

    # print(teacher.robustness(0.1))
    # Example usage:
    x = torch.linspace(0,1,1001)
    x_matrix = x.repeat(dim, 1).t()
    y = Sawtooth_wave().apply(x, 1)

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
    print(teacher.theoretical_robustness(t))

