import numpy as np
from independent_sampler.probabilities_transformer.exponential_twisting import ExponentialTwisting
from independent_sampler.probabilities_transformer.identity import IdentityTransformer

__author__ = 'daria'


class ImportanceSampling:
    """
    This class implements importance sampling strategy. It transform probabilities with transformer and
    perform importance sampling.
    """

    def __init__(self, transformer, n_iterations=1000):
        """
        Constructor
        :param transformer: transform the probability. It implements __call__ method and take probabilities and losses
        as input
        :return: new vector of probabilities.
        """
        self.transformer = transformer
        self.n_iterations = n_iterations
        np.random.seed(13)

    def _get_one_loss_new_prob(self, new_probabilities, probability_ratio, reverse_ratio, losses, threshold):
        """
        perform one simulation of important sampling and return normalized result. See Glasserman and Li
        Importance Sampling for Portfolio Credit Risk formula (3)

        :param new_probabilities: probabilities after transformation
        :param probability_ratio: ratio old_probabilities / new_probabilities
        :param reverse_ratio: (1 - old_probabilities) / (1 - new_probabilities)
        :param losses: np.array of losses
        :param threshold: the threshold of big loss
        :return: normalized result of one simulation
        """
        is_default = new_probabilities > np.random.ranf(new_probabilities.shape[0])
        loss = (losses * is_default).sum()
        return (loss > threshold) * np.prod(probability_ratio ** is_default) * np.prod(
            reverse_ratio ** (1 - is_default))

    def importance_sample(self, new_probabilities, probability_ratio, reverse_ratio, losses, threshold, n_iterations):
        """

        :param new_probabilities: probabilities after transformation
        :param probability_ratio: ratio old_probabilities / new_probabilities
        :param reverse_ratio: (1 - old_probabilities) / (1 - new_probabilities)
        :param losses: np.array of losses
        :param threshold: the threshold of big loss
        :param n_iterations:
        :return: estimation of probability of big loss
        """
        probability = 0
        for i in range(n_iterations):
            probability += self._get_one_loss_new_prob(new_probabilities, probability_ratio, reverse_ratio, losses,
                                                       threshold) / n_iterations
        return probability

    def sample(self, data, threshold):
        """
        estimate the probability of the losses greater than threshold
        :param data: list of tuple. tuple is a pair of (default probability, loss)
        :param threshold: the threshold of big loss
        :return: probability of the big loss
        """
        probabilities, losses = zip(*data)
        # print(probabilities)
        probabilities, losses = np.array(probabilities), np.array(losses)
        new_probabilities = self.transformer(probabilities, losses, threshold)

        ratio = probabilities / new_probabilities
        reverse_ratio = (1 - probabilities) / (1 - new_probabilities)
        return self.importance_sample(new_probabilities, ratio, reverse_ratio, losses, threshold, self.n_iterations)


if __name__ == "__main__":

    data = [(0.001, 1), (0.0009, 1)]

    print(ImportanceSampling(ExponentialTwisting(), 1000).sample(data, 1))
