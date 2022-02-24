import numpy as np
import timeit
start_time = timeit.default_timer()
np.sqrt(np.random.rand() ** 2 + np.random.rand() ** 2)
elapsed = timeit.default_timer() - start_time
print(elapsed)