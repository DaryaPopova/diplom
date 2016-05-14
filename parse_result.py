from statistics import mean, stdev

n = 0
baseline = []
important = []
big = []


baseline_time = []
important_time = []
big_time = []


for line in open("results"):
    if n % 9 == 3:
        baseline.append( int(line.split()[-2]))
    if n % 9 == 5:
        important.append(int(line.split()[-2]))
    if n % 9 == 7:
        big.append(int(line.split()[-2]))

    if n % 9 == 4:
        baseline_time.append(float(line.split()[-1]))
    if n % 9 == 6:
        important_time.append(float(line.split()[-1]))
    if n % 9 == 8:
        big_time.append(float(line.split()[-1]))
    n += 1


def get_result(data):
    return "{} +- {}".format(mean(data), stdev(data) / len(data) ** 0.5)

print(get_result(baseline), get_result(important), get_result(big))
print(get_result(baseline_time), get_result(important_time), get_result(big_time))
