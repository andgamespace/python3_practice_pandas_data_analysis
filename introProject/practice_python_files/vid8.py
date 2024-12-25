from statistics import mean 
import numpy as np

import matplotlib.pyplot as plt
xs = np.array([1,2,3,4,5,6])
ys = np.array([5,4,6,5,6,7])

def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    
    b = mean(ys) - m*mean(xs)
    
    return m, b
m, b = best_fit_slope_and_intercept(xs,ys)
print(m, b)
# plt.scatter(xs,ys)
# plt.show()
regression_line = [(m*x)+b for x in xs]
print(regression_line)

plt.scatter(xs,ys)
plt.plot(xs, regression_line)
plt.show()