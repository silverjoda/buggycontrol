import numpy as np
from scipy import interpolate
from matplotlib import pyplot as plt

x = np.array([23, 23.5, 24, 24, 25, 25])
y = np.array([13, 12.6, 12, 13, 12, 13])

# append the starting x,y coordinates
# x = np.r_[x, x[0]]
# y = np.r_[y, y[0]]

# fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
# is needed in order to force the spline fit to pass through all the input points.
# tck, u = interpolate.splprep([x, y], per=True, s=0)
tck, u = interpolate.splprep([x, y], s=0, k=3)
print(tck[0])
print(tck[1])
print(tck[2])
print(u)

# evaluate the spline fits for 1000 evenly spaced distance values
xi, yi = interpolate.splev(np.linspace(0, 1, 1000), tck)

# plot the result
fig, ax = plt.subplots(1, 1)
ax.plot(x, y, 'or')
ax.plot(xi, yi, '-b')
plt.show()