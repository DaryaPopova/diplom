from statistics import mean, stdev
from borrower import Borrower
import numpy as np
from scipy.stats import norm
from independent_sampler.importance_sampling import ImportanceSampling
from independent_sampler.independent_defaults import StandardSampling
from independent_sampler.probabilities_transformer.exponential_twisting import ExponentialTwisting

__author__ = 'daria'


class TwoStepSampler:

    def __init__(self, independent_sampler):
        self.independent_sampler = independent_sampler

    def sample(self, borrowers, threshold, n_iterations=1000, eps=0.0001, target=None):
        """

        :param borrowers: list of borrower (and information about them)
        :type borrowers: list[Borrower]

        :param threshold: big losses threshold
        :type threshold: float

        :param n_iterations: number of simulations
        :type n_iterations: int

        :return:
        """
        weights_matrix, independent_weight, losses, vitality = self.get_parameters(borrowers)
        res = []
        iteration = 0
        for iteration in range(n_iterations):
            res.append(self.one_loss(weights_matrix, independent_weight, losses, vitality, threshold))
            if iteration > 100 and target is not None and abs(target - mean(res)) < eps:
                break
            elif iteration > 100 and (max(res) - min(res)) / (iteration ** 0.5) < eps:
                break
        print("TwoStepSampler break after {} iterations".format(iteration))

        return mean(res)

    def get_parameters(self, borrowers):
        """

        :param borrowers: list of borrower (and information about them)
        :type borrowers: list[Borrower]
        """
        weights_matrix = np.matrix([borrower.weight for borrower in borrowers])
        independent_weight = np.array([borrower.independent_weight for borrower in borrowers])
        losses = np.array([borrower.loss for borrower in borrowers])
        vitality = np.array([borrower.vitality for borrower in borrowers])

        return weights_matrix, independent_weight, losses, vitality

    def one_loss(self, weights_matrix, independent_weight, losses, vitality, threshold):
        """

        :type weights_matrix: np.matrix
        :type independent_weight: np.array
        :type losses: np.array
        :type vitality: np.array
        :type threshold: float
        :return:
        """
        z_factor = np.random.normal(0, 1, weights_matrix.shape[1])

        independent_vitality = (vitality - np.asarray(weights_matrix.dot(z_factor)).T)[0]
        normalized_vitality = independent_vitality / independent_weight
        probabilities = 1.0 - norm.cdf(normalized_vitality)

        data = np.dstack((probabilities, losses))[0].tolist()
        return self.independent_sampler.sample(data, threshold)

if __name__ == "__main__":

    borrowers = [Borrower(1, 0, [0.9]), Borrower(1, 0, [0.9])]
    tss = TwoStepSampler(ImportanceSampling(ExponentialTwisting(), 100))
    print(tss.sample(borrowers, 1.5))
