import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Quaternion, Vector3
class IMUControl:
    def __init__(self):
        rospy.init_node('imu_control')

        # Initialize publisher for velocity commands
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
	
        # Initialize subscriber for IMU data
        rospy.Subscriber('/imu/data2', Imu, self.imu_callback)

        self.twist = Twist()

    def imu_callback(self, msg):
        # Example: Using roll angle for steering
        #roll = msg.orientation.x
        # Example: Using pitch angle for forward velocity
        #pitch = msg.orientation.y
        
        # Calculate velocities based on IMU data
        #forward_velocity = pitch * 0.1  # Adjust scaling factor as needed
        #steering_velocity = roll * 0.5  # Adjust scaling factor as needed
        print(self.twist)
        travel_time = 1#float(rospy.get_time()) - float(msg.header.stamp.secs) 
        f = msg.linear_acceleration.x * travel_time
        g = msg.linear_acceleration.y * travel_time
        h = msg.linear_acceleration.z * travel_time
        # Populate twist message with calculated velocities
        self.twist.linear = Vector3(f, g, h)
        #self.twist.linear.y = msg.linear_acceleration[1]
        i = msg.angular_velocity.x
        j = msg.angular_velocity.y
        k = msg.angular_velocity.z
        self.twist.angular = Vector3(i * travel_time, j*travel_time, k * travel_time)

        # Publish velocity command
        self.cmd_vel_pub.publish(self.twist)

if __name__ == '__main__':
    imu_control = IMUControl()
    rospy.spin()
