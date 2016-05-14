import numpy as np
from scipy.optimize import newton_krylov, anderson, minimize

__author__ = 'daria'


class ExponentialTwisting:
    """
    This class perform exponential twisting as described in Importance Sampling for Portfolio Credit Risk
    Paul Glasserman, Jingyi Li section 3
    """

    def __init__(self):
        pass

    def get_optimal_theta(self, probabilities, losses, threshold):
        """
        calculate optimal theta for exponential twisting according to equation (7) of article
        :param probabilities: default probabilities
        :param losses: the loss of the default
        :param threshold: big default threshold
        :return: optimal theta
        """
        def phi_derivative(theta):
            exp = np.exp(losses * theta)
            return (((probabilities * losses * exp) / (1 + probabilities * (exp - 1))).sum() - threshold) ** 2

        if phi_derivative(0) < 0.01:
            solution = minimize(phi_derivative, 0.5).x  # TODO change this solver
        else:
            solution = 0
        return solution

    def __twist_probability(self, theta, probabilities, losses):
        """
        twist probability according to formula (4) of article
        her comes the q norm
        :param theta: parameter of exponential twisting
        :param probabilities: default probabilities
        :param losses: the loss of the default
        :return: new probabilities
        """
        exp = np.exp(theta * losses)
        return probabilities * exp / (1 + probabilities * (exp - 1))

    def __call__(self, probabilities, losses, threshold):
        optimal_theta = self.get_optimal_theta(probabilities, losses, threshold)
        return self.__twist_probability(optimal_theta, probabilities, losses)
