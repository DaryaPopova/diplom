from math import sqrt
__author__ = 'daria'


class Borrower:

    def __init__(self, loss, vitality, weight):
        """

        :param loss: the loss in the case of the default of this borrower
        :type loss: float

        :param vitality: exposure to defaults of this borrower. Probability of the default for borrower
         p = Phi^{-1}(vitality)
        :type vitality: float

        :param weight: the weight[i] show how the factor Z_i influence on given borrower. See section 4 of the
            article Importance Sampling for Portfolio Credit Risk Importance Sampling for Portfolio Credit Risk for
            more details
        :type weight: list[float]

        :return: instance of class Borrower
        """
        self.loss = loss
        self.vitality = vitality
        self.weight = weight
        self.independent_weight = sqrt(1.0 - sum(map(lambda x: x ** 2, weight)))

