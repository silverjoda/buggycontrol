import numpy as np
import timeit

a = np.random.rand(1000,400,5)
start_time = timeit.default_timer()
#b = a.reshape((1000 * 400, 5))
c = a[3,1] * a[33,2]
elapsed = timeit.default_timer() - start_time
print(elapsed)