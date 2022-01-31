import torch as T
import pickle
import numpy as np
import os
import tf2_ros
from tf.transformations import *

class Odometrycalibrator:
    def __init__(self):
        list_of_dataset_dict_lists = self.load_datasets()
        self.episodes = len(self.list_of_dataset_dict_lists)
        self.calibrate_using_positional_constraints(list_of_dataset_dict_lists)

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

    def update_towards(self, val, target, tau):
        sign = val < target
        update_val = np.minimum(target - val, tau) * sign - np.minimum(val - target, tau) * (not sign)
        return val + update_val

    def calibrate_using_positional_constraints(self, list_of_dataset_dict_lists):
        # Non-optimizable params:
        dt = 0.005

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
        v_odom = T.tensor([0.0, 0.0], requires_grad=True)
        p_odom = T.tensor([0.0, 0.0], requires_grad=True)

        for d in list_of_dataset_dict_lists[0]:
            # Read sensor data
            wheel_speed, accel_imu, ori_imu = self.dict_to_arr_data(d)

            # Transform accel_imu to imu_zrp frame
            accel_imu_zrp = None

            # Transform accel_imu_zrp to bl_zrp

            # Update vel in bl_zrp using acceleration, wheel speed and tau
            v_bl_zrp[0] = self.update_towards(v_bl_zrp[0], wheel_speed * wheel_speed_scalar, x_vel_tau) + accel_bl_zrp[0] * dt
            v_bl_zrp[1] = self.update_towards(v_bl_zrp[1], 0., y_vel_tau) + accel_bl_zrp[1] * dt