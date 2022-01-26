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

    def init_ros(self):
        rospy.init_node("odometry_publisher")

        self.tfBuffer = tf2_ros.Buffer()
        self.tflistener = tf2_ros.TransformListener(self.tfBuffer)
        self.broadcaster = tf2_ros.TransformBroadcaster()

        self.wheel_speed_sub = subscriber_factory("/wheel_speed", Float64)
        self.dv_sub = subscriber_factory("/imu/dv", Vector3Stamped)
        self.quat_sub = subscriber_factory("/filter/quaternion", QuaternionStamped)

        time.sleep(0.5)

        # self.baselinktoimu = quaternion_from_euler(0, 0, np.pi / 2)
        # self.pos = np.array([0., 0., 0.])
        # self.v = np.zeros(3)
        # self.dvrate = 200
        # self.g = np.array([0, 0, 9.8]) / self.dvrate
        # self.gcounter = 0
        # self.decay_vec = np.array([0.0005] * 3)

    def loop(self):
        # Read current sensor values (wheel, accel, quat)
        wheel_speed, dv_imu, quat = self.read_sensor_values()

        # Transform imu accel to imu_zrp frame
        dv_zrp = None


        # Subtract g from imu_accel in imu_zrp frame
        # Transform clean imu_accel from imu_zrp to bl_zrp
        # Update vel in bl_zrp using acceleration, wheel speed and decay
        # Transform vel from bl_zrp to odom using quat in odom
        # Update pos in odom using vel in odom
        # Publish tf
        pass

    def gather(self):
        pass

    def read_sensor_values(self):
        with self.wheel_speed_sub.lock:
            while self.wheel_speed_sub.msg is None: pass
            wheel_speed = self.wheel_speed_sub.msg.data
        with self.dv_sub.lock:
            while self.dv_sub.msg is None: pass
            dv = self.dv_sub.msg.vector
        with self.quat_sub.lock:
            while self.quat_sub.msg is None: pass
            quat = self.quat_sub.msg.quaternion

        return wheel_speed, dv, quat

    def publish_base_links(self, rpy):
        """
        given full orientation of a robot make odom to stable_base_link
        transformation that has only yaw rotation and stable_base_link to base_link
        that has only roll and pitch rotation. publish both

        :param rpy: full orientation of a buggy as a quaternion
        """
        r, p, y = euler_from_quaternion(rpy)
        rp = numpytoquaternion(quaternion_from_euler(r, p, 0))
        y = numpytoquaternion(quaternion_from_euler(0, 0, y))

        stablebaselink = make_tf(frame=self.odom, child=self.stable_base_link, pos=self.pos.copy(), q=y)
        baselink = make_tf(frame=self.stable_base_link, child=self.base_link, pos=np.zeros(3), q=rp)
        self.broadcaster.sendTransform(stablebaselink)
        self.broadcaster.sendTransform(baselink)

    def orientationcallback(self, msg):
        """
        correct orientation data from a sensor by sensor to baselink transformation,
        make base and stable base links and publish it

        :param: message containing orientation of imu_link frame in the world frame
        """
        rpy = quaterniontonumpy(msg.quaternion)
        rpy = quaternion_multiply(rpy, self.baselinktoimu)
        self.publish_base_links(rpy=rpy)

    def updatevelocity(self, dv):
        """
        substract mean (g / dv rate) value in an imu frame from dv and
        update velocity in an odom frame by a new dv sample

        :param dv: delta linear velocity in an imu_link frame
        """
        try:
            t = self.tfBuffer.lookup_transform(self.odom, self.imu, rospy.Time(0))
            qm = quaternion_matrix(quaterniontonumpy(t.transform.rotation))[:3, :3]
            iqm = quaternion_matrix(invquaterniontonumpy(t.transform.rotation))[:3, :3]
            gdv = np.matmul(iqm, self.g)
            dv = vectror3tonumpy(dv) - gdv
            self.v += np.matmul(qm, dv)
            self.v = np.maximum(np.zeros(3), (self.v - self.decay_vec) * (self.v > 0)) + \
	             np.minimum(np.zeros(3), (self.v + self.decay_vec) * (self.v < 0)) 

        except Exception as e:
            print(e)

    def updateposition(self):
        """
        compute new position of a stable_base_link in an odom frame
        """
        self.pos[:2] += self.v[:2] * (1. / self.dvrate)

    def updateg(self, q, dv):
        """
        compute mean of a (g / dv rate) value in an odom frame
        given a new dv sample in an imu frame

        :param q: full orientation of an imu (roll, pitch, yaw in quaternion)
        :param dv: change in linear velocity in an imu_link frame
        """
        qm = quaternion_matrix(quaterniontonumpy(q))[:3, :3]
        dv = vectror3tonumpy(dv)
        gnow = np.matmul(qm, dv)
        self.g = (self.g * self.gcounter + gnow) / (self.gcounter + 1)
        self.g[:2] = 0
        self.gcounter += 1

    def gravitationcallback(self, msg):
        """
        lookup current odom to imu rotation and update g
        value with a new delta velocity vector

        :param msg: message containing velocity changes in an imu_link frame
        """
        try:
            t = self.tfBuffer.lookup_transform(self.odom, self.imu, rospy.Time(0))
            self.updateg(q=t.transform.rotation, dv=msg.vector)
        except Exception as e:
            print(e)
        if self.gcounter >= 100:
            self.gsub.unregister()

    def dvcallback(self, msg):
        """
        given a new dv sample compute odometry, i.e. update
        velocity in an odom frame and position of a robot in an odom frame

        :param msg: message containing velocity changes in an imu_link frame
        """
        self.updatevelocity(dv=msg.vector)
        self.updateposition()


if __name__ == "__main__":
    OdometryPublisher()
