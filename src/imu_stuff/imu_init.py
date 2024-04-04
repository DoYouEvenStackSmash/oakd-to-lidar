#!/usr/bin/python3
import rospy
from sensor_msgs.msg import Imu

class IMUPublisher:
    def __init__(self):
        self.imu_pub = rospy.Publisher("/imu/data", Imu, queue_size=10)
        self.imu_sub = rospy.Subscriber("/imu_oak", Imu, self.imu_callback)

    def imu_callback(self, imu_msg):
        # Your callback logic here
        #imu_msg.header.frame_id = "/imu/data2"
        # imu_msg.header.stamp = rospy.Time.now()
        # This function will be called whenever an IMU message is received
        self.publish(imu_msg)

    def publish(self, imu_msg):
        self.imu_pub.publish(imu_msg)

def main():
    rospy.init_node('imu_publisher')
    imu_publisher = IMUPublisher()
    rospy.spin()  # Keep the node running

if __name__ == '__main__':
    main()
