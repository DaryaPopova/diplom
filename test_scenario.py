import random

import time

from dependent_sampler.two_step_sampler import TwoStepSampler
from independent_sampler.big_loan_aware import BigLoanAwareSampler
from independent_sampler.importance_sampling import ImportanceSampling
from independent_sampler.independent_defaults import StandardSampling
from independent_sampler.probabilities_transformer.exponential_twisting import ExponentialTwisting
from preprocessing.covariation_calculator import build_borrower


def sample_sum():
    if random.uniform(0, 1) > 0.9862327:
        return random.gauss(50000, 1000)  # big borrower
    else:
        return random.gauss(10000, 1000)  # small borrower


def generate_input(number_of_borrower):
    """
    generate random input for sampling. Rating and sum is independent, sum of loan distributed uniformly.
    :param number_of_borrower:
    :return:
    """
    ratings = ["AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-", "CCC/C"]

    return [(random.choice(ratings), sample_sum()) for _ in range(number_of_borrower)]


def main():
    number_of_borrower = 50
    borrowers = build_borrower(generate_input(number_of_borrower))
    threshold = 45000
    print(threshold, max([borrower.loss for borrower in borrowers]), len([1 for b in borrowers if b.loss > threshold]))

    etalon = TwoStepSampler(StandardSampling(3500))

    start = time.time()
    true_score = etalon.sample(borrowers, threshold, 3500, eps=0.0)
    print("etalon", true_score, time.time() - start)
    baseline = TwoStepSampler(StandardSampling(n_iterations=350))
    eps = 0.0001
    start = time.time()
    print("baseline", baseline.sample(borrowers, threshold, 10000, eps=eps, target=true_score), time.time() - start)

    start = time.time()
    is_sampler = TwoStepSampler(ImportanceSampling(ExponentialTwisting(), n_iterations=350))
    print("is_sampler", is_sampler.sample(borrowers, threshold, 10000, eps=eps, target=true_score), time.time() - start)

    big_loan_sampler = TwoStepSampler(BigLoanAwareSampler(ImportanceSampling(ExponentialTwisting(), n_iterations=350)))
    start = time.time()
    print("big_loan_sampler", big_loan_sampler.sample(borrowers, threshold, 10000, eps=eps, target=true_score), time.time() - start)


for i in range(50):
    main()
