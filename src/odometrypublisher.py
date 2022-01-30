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
        self.imu_to_bl_quat = quaternion_from_euler(0, 0, np.pi / 2)
        self.x_vel_tau = 0.005
        self.y_vel_tau = 0.01
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
            g_cum += accel_imu[2]
            if ctr > 100:
                avg_g = g_cum / float(ctr)
                print("Current g is {}".format(avg_g))
                return avg_g
            self.ros_rate.sleep()

    def loop(self):
        # Take average of gravitational acceleration
        self.grav_accel = self.get_avg_g()

        while not rospy.is_shutdown():
            # Read current sensor values (wheel, imu, quat)
            ret = self.read_sensor_values()
            if ret is None:
                self.ros_rate.sleep()
                continue
            else:
                wheel_speed, accel_imu, quat_imu = ret

            # TODO: rename all quaternions to from_to_ notation for clarity
            # TODO: Continue correcting and checking with rviz

            # Transform imu accel to imu_zrp frame
            r, p, y = euler_from_quaternion(quat_imu)
            quat_imu_zrp = quaternion_from_euler(r, p, 0)
            accel_imu_zrp = self.rotate_vector_by_quat(accel_imu, quat_imu_zrp)

            # Subtract g from imu_accel in imu_zrp frame
            accel_imu_zrp[2] -= self.grav_accel

            # Transform (rotate) clean imu_accel from imu_zrp to bl_zrp
            quat_bl_zrp = quaternion_multiply(quat_imu_zrp, self.imu_to_bl_quat) # !Check with RVIZ if this is correct!
            accel_bl_zrp = self.rotate_vector_by_quat(accel_imu_zrp, self.imu_to_bl_quat)

            # Update vel in bl_zrp using acceleration, wheel speed and decay
            self.v_bl_zrp[0] = self.update_towards(self.v_bl_zrp[0], wheel_speed * self.wheel_speed_scalar, self.x_vel_tau) + accel_bl_zrp[0]
            self.v_bl_zrp[1] = self.update_towards(self.v_bl_zrp[1], 0, self.y_vel_tau) + accel_bl_zrp[1]

            # Transform vel from bl_zrp to odom using quat in odom
            _, _, bl_zrp_y = euler_from_quaternion(quat_bl_zrp)
            quat_yaw_odom = quaternion_from_euler(0, 0, bl_zrp_y)
            self.v_odom = self.rotate_vector_by_quat(self.v_bl_zrp, quat_yaw_odom)

            # Update pos in odom using vel in odom
            self.p_odom[0] += self.v_odom[0] * self.dt
            self.p_odom[1] += self.v_odom[1] * self.dt

            # Publish tfs
            odom_tf = make_tf(frame="odom", child="base_link_zrp", pos=[0,0,0], q=quat_yaw_odom)
            self.broadcaster.sendTransform(odom_tf)

            imu_zrp_tf = make_tf(frame="odom", child="imu ", pos=[0, 0, 0], q=quat_imu) # WRONG, need to transform by pi/2 first
            self.broadcaster.sendTransform(imu_zrp_tf)

            imu_zrp_tf = make_tf(frame="imu_zrp", child="imu", pos=[0,0,0], q=quat_imu_zrp)
            self.broadcaster.sendTransform(imu_zrp_tf)

            base_link_tf = make_tf(frame="base_link_zrp", child="base_link", pos=[0,0,0], q=quat_bl_zrp)
            self.broadcaster.sendTransform(base_link_tf)

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
