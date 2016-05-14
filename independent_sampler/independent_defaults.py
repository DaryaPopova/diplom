import numpy as np
from numpy.ma import mean

__author__ = 'daria'


class StandardSampling:

    def __init__(self, n_iterations=1000):
        """
        this is a standard sampler strategy for loss estimation.
        :return:
        """
        self.n_iterations = n_iterations

    def sample(self, data, threshold):
        """
        estimate the probability of the losses greater than threshold
        :param data: list of tuple. tuple is a pair of (default probability, loss)
        :param threshold: the threshold of big loss
        :param n_iterations: the number of sample
        :return: probability of the big loss
        """
        return mean(self.sample_losses(data, self.n_iterations) > threshold)

    def sample_losses(self, data, n_iterations=1000):
        """
        generate the sample of losses.
        :param data: list of tuple. tuple is a pair of (default probability, loss)
        :param n_iterations: the size of sample
        :return: list of default loss
        """
        probabilities, losses = zip(*data)
        probabilities, losses = np.array(probabilities), np.array(losses)

        return np.array([self._get_one_loss(probabilities, losses) for i in range(n_iterations)])

    def _get_one_loss(self, probabilities, losses):
        """
        perform one simulation and calculate the loss from the defaults
        :param probabilities: np.array of probabilities
        :param losses: np.array of losses
        :return: the loss in one simulation
        """
        is_default = probabilities > np.random.ranf(probabilities.shape[0])
        return (losses * is_default).sum()


if __name__ == "__main__":

    data = [(0.1, 1), (0.9, 1)]

    print(StandardSampling().sample(data, 1))
