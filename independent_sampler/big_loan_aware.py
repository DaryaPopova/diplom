from numpy import prod

from independent_sampler.importance_sampling import ImportanceSampling


class BigLoanAwareSampler:

    def __init__(self, sampler):
        """
        this sampler take into account that some borrowers loss greater than threshold
        :param sampler: sampler for estimate probabilities of big loss caused by default of small borrowers
        :type sampler: ImportanceSampling
        :return:
        """
        self.sampler = sampler

    def sample(self, data, threshold):
        """
        estimate the probability of the losses greater than threshold
        :param data: list of tuple. tuple is a pair of (default probability, loss)
        :param threshold: the threshold of big loss
        :param n_iterations: the number of sample
        :return: probability of the big loss
        """
        big_borrowers = [borrower for borrower in data if borrower[1] >= threshold]
        small_borrowers = [borrower for borrower in data if borrower[1] < threshold]
        probas_of_big_borrower_default = 1 - prod([(1 - proba) for (proba, loss) in big_borrowers])
        probas_of_small_borrower_default = self.sampler.sample(small_borrowers, threshold)
        return probas_of_big_borrower_default + (1 - probas_of_big_borrower_default) * probas_of_small_borrower_default


