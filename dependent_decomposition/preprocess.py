from math import sqrt
import numpy as np
from numpy.linalg import cholesky

__author__ = 'daria'


class Preprocessor:

    def __init__(self, n_features):
        pass

    def __call__(self, probability_matrix):
        """
        take probability matrix and transform into representation
        weight + free_weight: w * z + b * xi

        :param probability_matrix:
        :return:
        """

    @staticmethod
    def probability2correlation(probability_matrix):
        """
         We take the probability matrix and convert it to correlation matrix.
        :param probability_matrix: numpy array or numpy matrix
        :return: numpy matrix
        """
        shape = probability_matrix.shape
        correlation_matrix = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                p1 = probability_matrix[i, i]
                p2 = probability_matrix[j, j]
                joint_probability = probability_matrix[i, j]
                correlation_matrix[i, j] = Preprocessor.__calculate_correlation(p1, p2, joint_probability)
        return correlation_matrix

    @staticmethod
    def __calculate_correlation(p1, p2, joint_probability):
        return (joint_probability - p1 * p2) / sqrt(p1 * p2 * (1 - p1) * (1 - p2))

    def calculate_weights(self, matrix):
        decomposed_matrix = cholesky(self.probability2correlation(matrix))
        features_weight = sum(decomposed_matrix.T).tolist()[0]
        threshold = sorted(features_weight)[int(0.1 * len(features_weight))]
        to_take = [w > threshold for w in features_weight]
        weights = [self._get_weight(weight, to_take) for weight in decomposed_matrix.tolist()]
        return weights

    @staticmethod
    def _get_weight(weights, to_take):
        weight = [w for w, take in zip(weights, to_take) if take]
        free_weight = sum(weights) - sum(weight)
        return weight, free_weight
