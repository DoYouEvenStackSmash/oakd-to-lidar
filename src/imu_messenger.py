#!/usr/bin/python3
import rospy
from sensor_msgs.msg import LaserScan
import math
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3
import numpy as np


def populate_imu(packet):
    imu_msg = Imu()

    # Header
    imu_msg.header.stamp = rospy.Time.now()
    imu_msg.header.frame_id = "base_link"  # Adjust the frame_id according to your setup

    # Orientation
    rotation_vector = packet.rotationVector
    imu_msg.orientation = Quaternion(
        rotation_vector.i, rotation_vector.j, rotation_vector.k, rotation_vector.real
    )  # Set quaternion values (x, y, z, w)

    # Angular velocity
    gyroValues = packet.gyroscope
    imu_msg.angular_velocity = Vector3(
        gyroValues.z, gyroValues.y, gyroValues.x
    )  # Set angular velocity values (x, y, z)

    # Linear acceleration
    linear_acceleration = packet.acceleroMeter
    imu_msg.linear_acceleration = Vector3(
        linear_acceleration.z, linear_acceleration.y, linear_acceleration.x
    )
    return imu_msg


def send_to_imu(imu_pub, imuData):
    PACKET_COUNT
    packets = imuData.packets
    for i in range(min(len(packets), PACKET_COUNT)):
        packet = packets[i]
        imu_msg = populate_imu(packet)
        imu_pub.publish(imu_msg)
