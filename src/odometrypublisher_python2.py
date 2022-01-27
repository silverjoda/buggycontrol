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

class OdometryPublisher:
    def __init__(self):
        self.init_ros()
        self.define_calibration_params()
        self.loop()

    def init_ros(self):
        rospy.init_node("odometry_publisher")

        self.tfBuffer = tf2_ros.Buffer()
        self.tflistener = tf2_ros.TransformListener(self.tfBuffer)
        self.broadcaster = tf2_ros.TransformBroadcaster()

        self.wheel_speed_sub = subscriber_factory("/wheel_speed", Float64)
        self.dv_sub = subscriber_factory("/imu/dv", Vector3Stamped)
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
        self.dt = 0.005

    def loop(self):
        while not rospy.is_shutdown():
            # Read current sensor values (wheel, accel, quat)
            ret = self.read_sensor_values()
            if ret is None:
                self.ros_rate.sleep()
                continue
            else:
                wheel_speed, dv_imu, quat_imu = ret

            # Transform imu accel to imu_zrp frame
            r, p, y = euler_from_quaternion(quat_imu)
            quat_imu_zrp = quaternion_from_euler(r, p, 0)
            dv_imu_zrp = self.rotate_vector_by_quat(dv_imu, quat_imu_zrp)

            # Subtract g from imu_accel in imu_zrp frame
            dv_imu_zrp[2] -= self.grav_accel

            # Transform (rotate) clean imu_accel from imu_zrp to bl_zrp
            quat_bl_zrp = quaternion_multiply(quat_imu_zrp, self.imu_to_bl_quat)
            dv_bl_zrp = self.rotate_vector_by_quat(dv_imu_zrp, self.imu_to_bl_quat)

            # Update vel in bl_zrp using acceleration, wheel speed and decay
            self.v_bl_zrp[0] = self.update_towards(self.v_bl_zrp[0], wheel_speed, self.x_vel_tau) + dv_bl_zrp[0]
            self.v_bl_zrp[1] = self.update_towards(self.v_bl_zrp[1], 0, self.y_vel_tau) + dv_bl_zrp[1]

            # Transform vel from bl_zrp to odom using quat in odom
            quat_yaw_odom = quaternion_from_euler(0, 0, y)
            self.v_odom = self.rotate_vector_by_quat(self.v_bl_zrp, quat_yaw_odom)

            # Update pos in odom using vel in odom
            self.p_odom[0] += self.v_odom[0] * self.dt
            self.p_odom[1] += self.v_odom[1] * self.dt

            # Publish tf (bl, bl_zrp, imu_zrp)
            odom_tf = make_tf(frame="odom", child="base_link_zrp", pos=self.p_odom, q=quaternion_multiply(quat_yaw_odom, self.imu_to_bl_quat))
            self.broadcaster.sendTransform(odom_tf)

            bl_zrp_tf = make_tf(frame="base_link_zrp", child="base_link", pos=[0,0,0], q=quat_imu_zrp)
            self.broadcaster.sendTransform(bl_zrp_tf)

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
        with self.wheel_speed_sub.lock:
            if self.wheel_speed_sub.msg is None: return None
            wheel_speed = self.wheel_speed_sub.msg.data
        with self.dv_sub.lock:
            if self.dv_sub.msg is None: return None
            dv = vector3tonumpy(self.dv_sub.msg.vector)
        with self.quat_sub.lock:
            if self.quat_sub.msg is None: return None
            quat = quaterniontonumpy(self.quat_sub.msg.quaternion)

        return wheel_speed, dv, quat


if __name__ == "__main__":
    OdometryPublisher()
