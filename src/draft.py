#import torch as T
from tf.transformations import *
import numpy as np
T = None
#mat_tx_no_yaw = T.tensor([[1,0,0],[0,1,0],[0,0,1]], requires_grad=True)

max_iters = 1000
for i in range(max_iters):
    # generate random r,p,y
    r, p, y = np.random.rand(3) - 0.5

    # generate matrix from rpy and rp0
    mat_rpy = euler_matrix(r, p, y)[:3, :3]
    mat_rp0 = euler_matrix(r, p, 0)

    # generate rp0 matrix using y_n_mat
    mat_rp0_pred = T.matmul(mat_tx_no_yaw, T.from_numpy(mat_rpy))

    # loss
    loss = T.pow(T.from_numpy(mat_rp0) - mat_rp0_pred, 2)

    # backprop
    loss.backward()

    # apply grad
    with T.no_grad:
        mat_tx_no_yaw -= mat_tx_no_yaw.grad * 0.01

    # Zero grad
    mat_tx_no_yaw.grad.fill_(0)

    if i % 10 == 0:
        print("Iter: {}/{}, loss: {}".format(i, max_iters, loss.data))

print("Result:")
print(mat_tx_no_yaw)
print("Done.")




