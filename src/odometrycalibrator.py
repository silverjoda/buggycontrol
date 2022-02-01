import torch as T
import pickle
import numpy as np
import os
import tf2_ros
from tf.transformations import *
import cma

class Odometrycalibrator:
    def __init__(self):
        list_of_dataset_dict_lists = self.load_dataset()
        self.episodes = len(list_of_dataset_dict_lists)
        self.bbx_calibrate_using_positional_constraints(list_of_dataset_dict_lists)

    def load_dataset(self):
        # Load all datasets
        list_of_data_dict_lists = []
        dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/dataset")
        for i in range(100):
            file_path = os.path.join(dataset_dir, "dataset_{}.pkl".format(i))
            if os.path.exists(file_path):
                # list_of_data_dict_lists.append(pickle.load(open(file_path, "rb"), encoding='latin1'))
                list_of_data_dict_lists.append(pickle.load(open(file_path, "rb")))

                assert len(list_of_data_dict_lists[-1]) > 100
        assert len(list_of_data_dict_lists) > 0
        print("Loaded dataset with {} episodes".format(len(list_of_data_dict_lists)))
        return list_of_data_dict_lists

    def dict_to_arr_data(self, d):
        wheel_speed = d["/wheel_speed"].data
        imu_data = d["/imu/data"].linear_acceleration
        imu_data_arr = np.array([imu_data.x, imu_data.y, imu_data.z])
        quat_imu = d["/filter/quaternion"].quaternion
        mat_imu_arr = quaternion_matrix(quat_imu)[:3, :3]
        return T.tensor(wheel_speed), T.from_numpy(imu_data_arr), T.from_numpy(mat_imu_arr)

    def test_functions(self):
        r, p, y = np.random.rand(3) * 2 - 1.0

        # generate matrix from rpy and rp0
        mat_rpy = euler_matrix(r, p, y)[:3, :3].astype(np.float32)
        mat_rp0 = euler_matrix(r, p, 0)[:3, :3].astype(np.float32)

        T_mat_rpy = T.from_numpy(mat_rpy.astype(np.float32))
        T_mat_rp0 = self.remove_yaw_from_matrix_tensor(T_mat_rpy)

        print(mat_rp0)
        print(T_mat_rp0)

    def remove_yaw_from_matrix_tensor(self, m):
        yaw_mat = self.get_yaw_mat_from_matrix_tensor(m)
        no_yaw_mat = T.matmul(yaw_mat.inverse(), m)
        return no_yaw_mat

    def get_yaw_mat_from_matrix_tensor(self, m):
        # a, b, c, d, e, f
        # 0, 1, 2, 0, 1, 2
        # bf-ce, cd-af, ae-bd
        # 12-21, 20-02, 01-10

        # Project to plane
        proj_mat = T.matmul(m, T.tensor([[1., 0.], [0., 1.], [0., 0.]]))

        # Get individual vectors
        x = proj_mat[:, 0]
        y = proj_mat[:, 1]

        # Take cross product
        z = T.tensor([x[1] * y[2] - x[2] * y[1],
                      x[2] * y[0] - x[0] * y[2],
                      x[0] * y[1] - x[1] * y[0]])

        yaw_mat = T.cat([proj_mat, z.unsqueeze(1)], dim=1)

        return yaw_mat

    def differential_calibrate_using_positional_constraints(self, list_of_dataset_dict_lists):
        # Non-optimizable params:
        dt = 0.005

        # Lossfuns
        mse_loss = T.nn.MSELoss()

        # Define optimization vars, etc.
        wheel_speed_scalar = T.tensor(1.0, requires_grad=True)
        bl_zrp_to_imu_zrp_corr_mat = T.tensor([[1.,0.,0.],
                                               [0.,1.,0.],
                                               [0.,0.,1.]], requires_grad=True)

        x_vel_tau = T.tensor(0.002, requires_grad=True)
        y_vel_tau = T.tensor(0.003, requires_grad=True)
        x_accel_linear_correction = T.tensor([0.0, 1.0], requires_grad=True)
        y_accel_linear_correction = T.tensor([0.0, 1.0], requires_grad=True)
        z_accel_linear_correction = T.tensor([0.0, 1.0], requires_grad=True)

        # XY integrated vel and pos
        v_bl_zrp = T.tensor([0.0, 0.0], requires_grad=True)
        p_odom = T.tensor([0.0, 0.0], requires_grad=True)

        for d in list_of_dataset_dict_lists[0]:
            # Read sensor data
            wheel_speed, accel_imu, ori_imu = self.dict_to_arr_data(d)

            # Transform accel_imu to imu_zrp frame
            accel_imu_zrp = T.matmul(accel_imu, self.remove_yaw_from_matrix_tensor(ori_imu))

            # Transform accel_imu_zrp to bl_zrp
            accel_bl_zrp = T.matmul(bl_zrp_to_imu_zrp_corr_mat, accel_imu_zrp)

            # Update vel in bl_zrp using acceleration, wheel speed and tau
            v_bl_zrp[0] = self.update_towards(v_bl_zrp[0], wheel_speed * wheel_speed_scalar, x_vel_tau) + accel_bl_zrp[0] * dt
            v_bl_zrp[1] = self.update_towards(v_bl_zrp[1], 0., y_vel_tau) + accel_bl_zrp[1] * dt

            v_odom = T.matmul(self.get_yaw_mat_from_matrix_tensor(ori_imu), v_bl_zrp)
            p_odom += v_odom * dt

        # Define losses
        pos_loss = mse_loss(p_odom[:2], T.tensor([0., 0.]))
        decay_losses = T.pow(x_vel_tau, 2) * 0.01 + T.pow(y_vel_tau, 2) * 0.01
        loss_sum = pos_loss + decay_losses
        loss_sum.backward()

        # Apply grads
        wheel_speed_scalar -= wheel_speed_scalar.grad * 0.01
        bl_zrp_to_imu_zrp_corr_mat -= bl_zrp_to_imu_zrp_corr_mat.grad * 0.01
        x_vel_tau -= x_vel_tau.grad * 0.01
        y_vel_tau -= y_vel_tau.grad * 0.01

        # Zero grads
        wheel_speed_scalar.grad.fill_(0)
        bl_zrp_to_imu_zrp_corr_mat.grad.fill_(0)
        x_vel_tau.grad.fill_(0)
        y_vel_tau.grad.fill_(0)

    def update_towards(self, val, target, tau):
        sign = val < target
        update_val = np.minimum(target - val, tau) * sign - np.minimum(val - target, tau) * (not sign)
        return val + update_val

    def rotate_vector_by_quat(self, v, q):
        qm = quaternion_matrix(q)[:3, :3]
        return np.matmul(qm, v)

    def bbx_objective(self, w, d_l, v_bl_zrp, p_odom):
        dt = 0.005
        wheel_speed_scalar = w[0]
        bl_zrp_to_imu_zrp_corr_mat = quaternion_from_matrix(np.array(w[1:10]).reshape(3,3))
        x_vel_tau = w[10]
        y_vel_tau = w[11]

        for d in d_l:
            # Read sensor data
            wheel_speed = d["/wheel_speed"].data
            imu_data = d["/imu/data"].linear_acceleration
            accel_imu = np.array([imu_data.x, imu_data.y, imu_data.z])
            quat_imu = d["/filter/quaternion"].quaternion
            q_imu = [quat_imu.x, quat_imu.y, quat_imu.z, quat_imu.w]

            # Transform imu accel to imu_zrp frame
            r_imu_imuinit, p_imu_imuinit, y_imu_imuinit = euler_from_quaternion(q_imu)
            q_imu_imuzrp = quaternion_from_euler(r_imu_imuinit, p_imu_imuinit, 0)
            accel_imu_zrp = self.rotate_vector_by_quat(accel_imu, q_imu_imuzrp)

            # Transform (rotate) imu_accel from imu_zrp to bl_zrp
            accel_bl_zrp = self.rotate_vector_by_quat(accel_imu_zrp, bl_zrp_to_imu_zrp_corr_mat)

            # Update vel in bl_zrp using acceleration, wheel speed and decay #
            v_bl_zrp[0] = self.update_towards(v_bl_zrp[0], wheel_speed * wheel_speed_scalar, x_vel_tau) + accel_bl_zrp[0] * dt
            v_bl_zrp[1] = self.update_towards(v_bl_zrp[1], 0., y_vel_tau) + accel_bl_zrp[1] * dt

            # Transform vel from bl_zrp to odom using quat in odom
            q_blzrp_odom = quaternion_from_euler(0, 0, y_imu_imuinit)
            v_odom = self.rotate_vector_by_quat(v_bl_zrp, q_blzrp_odom)

            # Update pos in odom using vel in odom
            p_odom += v_odom * dt

        # Define losses
        pos_loss = np.mean(np.square(p_odom[:2], np.array([0., 0.])))
        decay_losses = np.square(x_vel_tau) * 0.01 + np.square(y_vel_tau) * 0.01
        loss_sum = pos_loss + decay_losses
        return loss_sum

    def bbx_calibrate_using_positional_constraints(self, list_of_dataset_dict_lists):
        # Optimizable parms
        wheel_speed_scalar = 1.0
        bl_zrp_to_imu_zrp_corr_mat = np.array([[1.,0.,0.],
                                               [0.,1.,0.],
                                               [0.,0.,1.]])
        x_vel_tau = 0.002
        y_vel_tau = 0.003

        w = []
        w.append(wheel_speed_scalar)
        w.extend(list(bl_zrp_to_imu_zrp_corr_mat.reshape(9)))
        w.append(x_vel_tau)
        w.append(y_vel_tau)

        # XY integrated vel and pos
        v_bl_zrp = np.array([0.0, 0.0])
        p_odom = np.array([0.0, 0.0])

        es = cma.CMAEvolutionStrategy(w, 0.03)
        es.optimize(lambda x: self.bbx_objective(x, list_of_dataset_dict_lists[0], v_bl_zrp, p_odom))

if __name__=="__main__":
    Odometrycalibrator()