from math import sqrt
import numpy as np
from numpy import genfromtxt
__author__ = 'daria'


def read_matrix(path):
    return genfromtxt(path, delimiter=',')


def __calculate_correlation(p1, p2, joint_probability):
    return (joint_probability - p1 * p2) / sqrt(p1 * p2 * (1 - p1) * (1 - p2))


def probability2correlation(probability_matrix):
    """
     We take the probability matrix and convert it to correlation matrix.
    :param probability_matrix: numpay array or numpy matrix
    :return: numpy matrix
    """
    shape = probability_matrix.shape
    correlation_matrix = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            p1 = probability_matrix[i, i]
            p2 = probability_matrix[j, j]
            joint_probability = probability_matrix[i, j]
            correlation_matrix[i, j] = __calculate_correlation(p1, p2, joint_probability)
    return correlation_matrix


