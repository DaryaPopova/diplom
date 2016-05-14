

"""
    Calculate default probabilities for pair of borrower
"""
from statistics import mean
from scipy.stats import norm
import pandas
import numpy as np
from numpy.linalg import cholesky

from borrower import Borrower

path = "~/Dropbox/edu/diplom/МатрицаПопарныхВероятностей/FirstTableAA.csv"
data = pandas.read_csv(path, index_col="YEAR")


def get_probas(data):
    ratings = list(data.columns.values)
    probas = {}
    for rating in ratings:
        probas[rating] = mean(data[rating]) * 0.01
    return probas


def get_pair_probas(data):
    raitings = list(data.columns.values)
    probas = {}
    for raiting in raitings:
        for raiting2 in raitings:
            probas[(raiting, raiting2)] = mean(data[raiting] * data[raiting2]) * 1e-4
    return probas


def build_borrower(borrowers):
    """

    :param borrowers: list of borrowers (rating, sum of losses)
    :type borrowers: list[(str, float)]
    :return:
    """
    probas = get_probas(data)
    pair_probas = get_pair_probas(data)
    shape = (len(borrowers), len(borrowers))
    cov_matrix = np.zeros(shape)
    for i in range(len(borrowers)):
        for j in range(len(borrowers)):
            rating1 = borrowers[i][0]
            rating2 = borrowers[j][0]
            cov_matrix[i, j] = pair_probas[(rating1, rating2)] - probas[rating1] * probas[rating2]
    cov_matrix = cov_matrix + np.eye(len(cov_matrix)) * 1e-6
    weight_matrix = cholesky(cov_matrix)
    return [Borrower(borrower[1], norm.isf(probas[borrower[0]]), weight) for borrower, weight in zip(borrowers, weight_matrix)]

if __name__ == "__main__":
    probas = get_probas(data)
    res = ""
    for i in ["AA-", "A", "BBB", "BB+", "B", "B-", "CCC/C"]:
        res += i + " &"
    res = res[:-1] + "\\\\\n"
    for i in ["AA-", "A", "BBB", "BB+", "B", "B-", "CCC/C"]:
        res += str(probas[i]) + " &"
    print(res)

    pair_probas = get_pair_probas(data)

    res = "\\hline\n"
    for i in [('A', 'BBB-'), ('AA-', 'A-'), ('A', 'B')]:
        res += str(i) + " &"
    res = res[:-1] + "\\\\\n\\hline\n"

    for i in [('A', 'BBB-'), ('AA-', 'A-'), ('A', 'B')]:
        res += str(pair_probas[i]) + " &"
    res = res[:-1] + "\n\\hline"
    print(res)


