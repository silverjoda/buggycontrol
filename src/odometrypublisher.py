#!/usr/bin/env python
import rospy
import tf2_ros
from tf.transformations import *
from geometry_msgs.msg import Vector3Stamped, QuaternionStamped, TransformStamped, Quaternion, Vector3
from std_msgs.msg import Float64
from utils_python2 import *
import numpy as np
import threading
import time
from sensor_msgs.msg import Imu

class OdometryPublisher:
    def __init__(self):
        self.init_ros()
        self.define_calibration_params()
        self.loop()

    def init_ros(self):
        rospy.init_node("odometry_publisher")
        print("Starting odometry publisher node...")

        self.tfBuffer = tf2_ros.Buffer()
        self.tflistener = tf2_ros.TransformListener(self.tfBuffer)
        self.broadcaster = tf2_ros.TransformBroadcaster()

        self.wheel_speed_sub = subscriber_factory("/wheel_speed", Float64)
        self.dv_sub = subscriber_factory("/imu/dv", Vector3Stamped)
        self.imu_sub = subscriber_factory("/imu/data", Imu)
        self.quat_sub = subscriber_factory("/filter/quaternion", QuaternionStamped)

        self.v_bl_zrp = [0,0,0]
        self.v_odom = [0,0,0]
        self.p_odom = [0,0,0]

        self.ros_rate = rospy.Rate(200)
        time.sleep(0.5)

    def define_calibration_params(self):
        self.grav_accel = 9.81
        self.imu_to_bl_quat = quaternion_from_euler(0, 0, -np.pi / 2)
        self.x_vel_tau = 0.002
        self.y_vel_tau = 0.002
        self.wheel_speed_scalar = 1.0
        self.dt = 0.005

    def get_avg_g(self):
        ctr = 0
        g_cum = 0
        while not rospy.is_shutdown():
            # Read current sensor values (wheel, imu, quat)
            ret = self.read_sensor_values()
            if ret is None:
                self.ros_rate.sleep()
                continue
            else:
                wheel_speed, accel_imu, quat_imu = ret

            ctr +=1
            g_cum += sum(accel_imu)
            if ctr > 100:
                avg_g = g_cum / float(ctr)
                print("Current g is {}".format(avg_g))
                return avg_g
            self.ros_rate.sleep()

    def loop(self):
        # Take average of gravitational acceleration
        #self.grav_accel = self.get_avg_g()
        self.grav_accel = 9.81

        while not rospy.is_shutdown():
            # Read current sensor values (wheel, imu, quat)
            ret = self.read_sensor_values()
            if ret is None:
                self.ros_rate.sleep()
                continue
            else:
                wheel_speed, accel_imu, q_imu_imuinit = ret

            # Transform imu accel to imu_zrp frame
            r_imu_imuinit, p_imu_imuinit, y_imu_imuinit = euler_from_quaternion(q_imu_imuinit)
            q_imu_imuzrp = quaternion_from_euler(r_imu_imuinit, p_imu_imuinit, 0)
            accel_imu_zrp = self.rotate_vector_by_quat(accel_imu, q_imu_imuzrp)

            # Transform (rotate) clean imu_accel from imu_zrp to bl_zrp
            q_imuzrp_blzrp = self.imu_to_bl_quat
            accel_bl_zrp = self.rotate_vector_by_quat(accel_imu_zrp, q_imuzrp_blzrp)

            # Update vel in bl_zrp using acceleration, wheel speed and decay #
            self.v_bl_zrp[0] = self.update_towards(self.v_bl_zrp[0], wheel_speed * self.wheel_speed_scalar, self.x_vel_tau) + accel_bl_zrp[0] * self.dt
            self.v_bl_zrp[1] = self.update_towards(self.v_bl_zrp[1], 0, self.y_vel_tau) + accel_bl_zrp[1] * self.dt

            # Transform vel from bl_zrp to odom using quat in odom
            q_blzrp_odom = quaternion_from_euler(0, 0, y_imu_imuinit)
            self.v_odom = self.rotate_vector_by_quat(self.v_bl_zrp, q_blzrp_odom)

            # Update pos in odom using vel in odom
            self.p_odom[0] += self.v_odom[0] * self.dt
            self.p_odom[1] += self.v_odom[1] * self.dt

            # Publish tfs
            odom_tf = make_tf(frame="odom", child="base_link_zrp", pos=self.p_odom, q=q_blzrp_odom)
            self.broadcaster.sendTransform(odom_tf)

            imu_zrp_tf = make_tf(frame="base_link_zrp", child="imu_zrp", pos=[-0.1,0,0], q=self.imu_to_bl_quat)
            self.broadcaster.sendTransform(imu_zrp_tf)

            imu_zrp_tf = make_tf(frame="imu_zrp", child="imu", pos=[0,0,0], q=q_imu_imuzrp)
            self.broadcaster.sendTransform(imu_zrp_tf)

            self.ros_rate.sleep()

    def update_towards(self, val, target, tau):
        sign = val < target
        update_val = np.minimum(target - val, tau) * sign - np.minimum(val - target, tau) * (not sign)
        return val + update_val

    def rotate_vector_by_quat(self, v, q):
        qm = quaternion_matrix(q)[:3, :3]
        return np.matmul(qm, v)

    def gather(self):
        pass

    def read_sensor_values(self):
        msg = self.wheel_speed_sub.get_msg(copy_msg=True)
        if msg is None: return None
        wheel_speed = self.wheel_speed_sub.msg.data

        msg = self.imu_sub.get_msg(copy_msg=True)
        if msg is None: return None
        imu_accel = vector3tonumpy(self.imu_sub.msg.linear_acceleration)

        msg = self.quat_sub.get_msg(copy_msg=True)
        if msg is None: return None
        quat = quaterniontonumpy(self.quat_sub.msg.quaternion)

        return wheel_speed, imu_accel, quat


if __name__ == "__main__":
    OdometryPublisher()
