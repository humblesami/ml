import numpy
from scipy import stats

speed = [99,86,87,88,111,90,103,77, 85, 91,94,78,85,89]
# speed = [86,87,88,86,87,85,86]
x = numpy.std(speed)
print(f'std={x}')

x = stats.mode(speed)
cnt = int(x.count)
mode = float(x.mode)
if int(mode) == mode:
    mode = int(mode)
print(f'mode={mode} with counts = {cnt}')

x = numpy.median(speed)
print(f'median={x}')

x = numpy.mean(speed)
print(f'mean={x}')

x = numpy.var(speed)
print(f'variance={x}')

ages = [5,31,43,48,50,41,7,11,15,39,80,82,32,2,8,6,25,36,27,61,31]
x = numpy.percentile(ages, 90)
print(f'percentile={x}')

x = numpy.random.uniform(0.0, 2.0, 7)
print(f'random_uniform={x}')

import matplotlib.pyplot as plt

x = numpy.random.uniform(0.0, 5.0, 250)
plt.hist(x, 5)
plt.show()

x = numpy.random.normal(5.0, 1.0, 100000)

plt.hist(x, 100)
plt.show()

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

plt.scatter(x, y)
plt.show()