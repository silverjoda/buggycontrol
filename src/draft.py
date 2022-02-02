import torch as T
from tf.transformations import *
import numpy as np

r, p, y = 0., np.pi / 4., 0,
acc_vec = np.array([-7,0,7.])

# generate matrix from rpy and rp0
mat_rpy = euler_matrix(r, p, y)[:3, :3].astype(np.float32)
mat_rp0 = euler_matrix(r, p, 0)[:3, :3].astype(np.float32)

acc_vec_zrp = np.matmul(mat_rp0, acc_vec)
print(acc_vec_zrp)




