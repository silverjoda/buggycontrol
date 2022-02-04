#!/usr/bin/env python3
import rospy
import tf2_ros
from tf.transformations import *
from geometry_msgs.msg import Vector3Stamped, QuaternionStamped, TransformStamped, Quaternion, Vector3
from std_msgs.msg import Float64
from utils import *
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
        self.accel_sub = subscriber_factory("/imu/acceleration", Vector3Stamped)
        self.quat_sub = subscriber_factory("/filter/quaternion", QuaternionStamped)

        self.v_bl_zrp = [0,0,0]
        self.v_odom = [0,0,0]
        self.p_odom = [0,0,0]

        self.ros_rate = rospy.Rate(200)
        time.sleep(0.5)

    def define_calibration_params(self):
        self.bl_to_imu = quaternion_from_euler(0, 0, 0)
        self.x_vel_tau = 0.01
        self.y_vel_tau = 1.
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
        while not rospy.is_shutdown():
            # Read current sensor values (wheel, imu, quat)
            ret = self.read_sensor_values()
            if ret is None:
                self.ros_rate.sleep()
                continue
            else:
                wheel_speed, accel_imu, q_imu_imuinit = ret

            if not hasattr(self, "prev_imu_accel_stamp"):
                dt = self.dt
            else:
                dt = self.get_dt(self.imu_accel_stamp, self.prev_imu_accel_stamp)
            self.prev_imu_accel_stamp = self.imu_accel_stamp

            # Transform imu accel to imu_zrp frame
            r_imu_imuinit, p_imu_imuinit, y_imu_imuinit = euler_from_quaternion(q_imu_imuinit)
            q_imu_imuzrp = quaternion_from_euler(r_imu_imuinit, p_imu_imuinit, 0)
            accel_imu_zrp = self.rotate_vector_by_quat(accel_imu, q_imu_imuzrp)

            # Transform (rotate) imu_accel from imu_zrp to bl_zrp
            #accel_bl_zrp = self.rotate_vector_by_quat(accel_imu_zrp, self.bl_to_imu)
            accel_bl_zrp = accel_imu_zrp

            # Update vel in bl_zrp using acceleration, wheel speed and decay #
            self.v_bl_zrp[0] = self.update_towards(self.v_bl_zrp[0], 0., self.x_vel_tau * dt) + accel_bl_zrp[0] * dt
            self.v_bl_zrp[1] = self.update_towards(self.v_bl_zrp[1], 0., self.y_vel_tau * dt) + accel_bl_zrp[1] * dt

            # Transform vel from bl_zrp to odom using quat in odom
            q_blzrp_odom = quaternion_from_euler(0, 0, y_imu_imuinit)
            self.v_odom = self.rotate_vector_by_quat(self.v_bl_zrp, q_blzrp_odom)

            #self.v_odom[0] += (accel_imu[0] * dt)
            #self.v_odom[1] += (accel_imu[1] * dt)

            # Update pos in odom using vel in odom
            self.p_odom[0] += (self.v_odom[0] * dt)
            self.p_odom[1] += (self.v_odom[1] * dt)

            # Publish tfs
            odom_tf = make_tf(frame="odom", child="base_link_zrp", pos=self.p_odom, q=q_blzrp_odom)
            self.broadcaster.sendTransform(odom_tf)

            imu_zrp_tf = make_tf(frame="base_link_zrp", child="imu_zrp", pos=[-0.1,0,0], q=self.bl_to_imu)
            self.broadcaster.sendTransform(imu_zrp_tf)

            imu_tf = make_tf(frame="imu_zrp", child="imu", pos=[0,0,0], q=q_imu_imuzrp)
            self.broadcaster.sendTransform(imu_tf)

            self.ros_rate.sleep()

    def update_towards(self, val, target, tau):
        if abs(tau) < 0.000001:
            return val
        if val < target:
            return np.minimum(val + tau, target)
        elif val > target:
            return np.maximum(val - tau, target)
        else:
            return val

    def rotate_vector_by_quat(self, v, q):
        qm = quaternion_matrix(q)[:3, :3]
        return np.matmul(qm, v)

    def gather(self):
        pass

    def get_dt(self, stamp_new, stamp_old):
        dt = stamp_new.secs - stamp_old.secs
        dt += (stamp_new.nsecs - stamp_old.nsecs) / 1000000000.
        return dt

    def read_sensor_values(self):
        msg = self.wheel_speed_sub.get_msg(copy_msg=True)
        if msg is None: return None
        wheel_speed = msg.data

        msg = self.accel_sub.get_msg(copy_msg=True)
        if msg is None: return None
        imu_accel = vector3tonumpy(msg.vector)
        self.imu_accel_stamp = msg.header.stamp

        msg = self.quat_sub.get_msg(copy_msg=True)
        if msg is None: return None
        quat = quaterniontonumpy(msg.quaternion)

        return wheel_speed, imu_accel, quat

if __name__ == "__main__":
    OdometryPublisher()
