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