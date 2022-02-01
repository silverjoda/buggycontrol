import torch as T
from tf.transformations import *
import numpy as np
mat_tx_no_yaw = T.rand(3, 3, requires_grad=True)

for i in range(10):
    # generate random r,p,y
    r, p, y = np.random.rand(3) * 2 - 1.0

    # generate matrix from rpy and rp0
    mat_rpy = euler_matrix(r, p, y)[:3, :3].astype(np.float32)
    mat_rp0 = euler_matrix(r, p, 0)[:3, :3].astype(np.float32)

    T_mat = np.matmul(mat_rp0, np.linalg.inv(mat_rpy))
    print(T_mat)



