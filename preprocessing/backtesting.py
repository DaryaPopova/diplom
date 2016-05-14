from statistics import mean

import pandas
import numpy as np
import matplotlib.pyplot as plt
from numpy.ma import std

path = "~/Dropbox/edu/diplom/МатрицаПопарныхВероятностей/FirstTableAA.csv"
data = pandas.read_csv(path, index_col="YEAR")


def get_upper(history_data):
    sorted_defaults = np.sort(history_data)
    return sorted_defaults[len(sorted_defaults) - int(len(sorted_defaults) * 0.05) - 1]


def get_lower(history_data):
    sorted_defaults = np.sort(history_data)
    return sorted_defaults[int(len(sorted_defaults) * 0.05)]


def is_hit(history_data, test_data, year):
    """

    :param history_data: probability for default given rating for a past years
    :type history_data: np.array
    :param test_data: default for given rating in given year
    :type test_data: float
    :return:
    """
    m = mean(history_data)
    st_dev = std(history_data)
    floar = get_upper(history_data)
    min_defaults = max(0.0, m - 1.96 * st_dev)
    max_defaults = min(100.0, m + 1.96 * st_dev)

    if year in [2007, 2008, 2011, 2014]:
        in_interval = "\\in"
        color = "green"
        if not min_defaults <= test_data <= max_defaults:
            in_interval = "\\not " + in_interval
            color = "red"
        min_defaults = "%.1f" % min_defaults
        max_defaults = "%.1f" % max_defaults
        end = "& " if year != 2014 else "\\\\"
        print("$\\textcolor{{{5}}}{{ {3} {4} [{1}, {2}] }}$".format(year, min_defaults, max_defaults, test_data, in_interval, color), end=end)
    if abs(test_data - m) <= 1.96 * st_dev:
        return 0
    return 1

values = list(data.columns.values)

for value in values:
    start_year = 19
    bb = data[value].tolist()
    upper = []
    lower = []
    hit = []
    years = np.arange(1981, 2016)

    print(value, end="& ")
    for i in range(start_year, len(bb)):
        upper.append(get_upper(bb[:i]))
        lower.append(get_lower(bb[:i]))
        if is_hit(bb[:i], bb[i], years[i]) == 1:
            hit.append(years[i])
    """
    plt.xticks(years)
    plt.plot(years[start_year + 1:], upper, label="upper")
    plt.plot(years[start_year + 1:], lower, label="lower")
    plt.plot(years[start_year + 1:], bb[start_year + 1:], "-o", label="test")
    plt.legend(loc=4)
    plt.title(value + " n_hit = {}".format(len(hit)))
    plt.show()
    """
    print("\n\hline")
    # print(value, hit)

