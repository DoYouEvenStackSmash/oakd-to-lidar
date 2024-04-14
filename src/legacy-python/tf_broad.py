#!/usr/bin/env python

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Imu
import tf.transformations as tf_trans
from geometry_msgs.msg import Quaternion
import math


def normalize_quaternion(quaternion_msg):
    # Compute the magnitude of the quaternion

    magnitude = math.sqrt(
        quaternion_msg.x**2
        + quaternion_msg.y**2
        + quaternion_msg.z**2
        + quaternion_msg.w**2
    )
    magnitude = 1
    # Normalize each component
    normalized_x = quaternion_msg.x / magnitude
    normalized_y = quaternion_msg.y / magnitude
    normalized_z = quaternion_msg.z / magnitude
    normalized_w = quaternion_msg.w / magnitude

    # Create a new normalized quaternion message
    normalized_quaternion_msg = Quaternion(
        normalized_x, normalized_y, normalized_z, normalized_w
    )

    return normalized_quaternion_msg


class IMUToOdometryTF:
    def __init__(self):
        rospy.init_node("imu_to_odometry_tf", anonymous=True)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        self.imu_sub = rospy.Subscriber("/imu", Imu, self.imu_callback)

    def imu_callback(self, imu_msg):
        # Convert IMU orientation to quaternion
        quaternion = normalize_quaternion(imu_msg.orientation)
        print("Firing")
        # Create a TransformStamped message
        tf_msg = TransformStamped()
        tf_msg.header.stamp = rospy.Time.now()
        tf_msg.header.frame_id = "odom"
        tf_msg.child_frame_id = "base_link"
        tf_msg.transform.translation.x = 0.0
        tf_msg.transform.translation.y = 0.0
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation = quaternion

        # Publish the TF message
        self.tf_broadcaster.sendTransform(tf_msg)

    def run(self):
        rospy.spin()  # Keep the node running


if __name__ == "__main__":
    imu_to_odometry_tf = IMUToOdometryTF()
    imu_to_odometry_tf.run()
