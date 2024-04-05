#!/usr/bin/python3
import numpy as np
import cv2
import depthai as dai
# import matplotlib.pyplot as plt
import sys
sys.path.append(".")
from viz import *
import matrix_processing as matrix_processing
from outlier_filter import *
from messenger import *

# create pipeline
pp = dai.Pipeline()
imu = pp.create(dai.node.IMU)

# create monocams because stereo wants them
left_lens = pp.createMonoCamera()
right_lens = pp.createMonoCamera()

# set resolution to 400p to save space
left_lens.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right_lens.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

# set FPS
left_lens.setFps(45)
right_lens.setFps(45)

# create stereo depth
stereo = pp.createStereoDepth()

# some stereo config stuff
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)
stereo.initialConfig.setConfidenceThreshold(220)
stereo.setRectifyEdgeFillColor(0)
stereo.setLeftRightCheck(False)
rgbCamSocket = dai.CameraBoardSocket.CAM_B
stereo.setDepthAlign(rgbCamSocket)

xoutImu = pp.createXLinkOut()
xoutDepth=pp.createXLinkOut()
xoutLeft=pp.createXLinkOut()
xoutRight = pp.createXLinkOut()

xoutImu.setStreamName("imu")
xoutLeft.setStreamName("left")
xoutRight.setStreamName("right")
xoutDepth.setStreamName("depth")


imu_q_size = 10
OAK_DS2 = False
OAKD = True

if OAK_DS2:
	imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, imu_q_size)
if OAKD:
	imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_CALIBRATED, imu_q_size)
	imu.enableIMUSensor(dai.IMUSensor.MAGNETOMETER_RAW, imu_q_size)
	imu.enableIMUSensor(dai.IMUSensor.LINEAR_ACCELERATION, imu_q_size)
	imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, imu_q_size)

imu.setBatchReportThreshold(1)
imu.setMaxBatchReports(10)
# Link plugins IMU -> XLINK
imu.out.link(xoutImu.input)
#
left_lens.setBoardSocket(dai.CameraBoardSocket.CAM_B)
right_lens.setBoardSocket(dai.CameraBoardSocket.CAM_C)
#
left_lens.out.link(stereo.left)
right_lens.out.link(stereo.right)
#
stereo.disparity.link(xoutDepth.input)


ROS = True
RATE = 12
DISPLAY = True
CUTOFF = 160

def main():
  if DISPLAY:
  	cv2.startWindowThread()
		cv2.namedWindow("scatter")
		cv2.startWindowThread()
		cv2.namedWindow("raw")
 
	val = np.zeros((400,640))+10000
	prev_markers = [np.array([10000 for _ in range(640)])]
	rolling_frame = [val]
	if ROS:
		rospy.init_node("populate_laserscan",anonymous=True)
		
  # Create a publisher
		laserscan_pub = rospy.Publisher("/scan", LaserScan, queue_size=10)
		imu_pub = rospy.Publisher("/imu_oak", Imu, queue_size=10)

		# Set the loop rate
		rate = rospy.Rate(60)  # 10 Hz
	FIRST=5
	with dai.Device(pp) as device:
		qdepth = device.getOutputQueue(name="depth",maxSize=30,blocking=False)
		imuQueue = device.getOutputQueue(name="imu", maxSize=10, blocking=False)
		c = 0
		numerator = 882.5 * 7.5	# focal point * baseline for OAK-D
		
		while not rospy.is_shutdown()and True:
			# nonblocking try to get frames
			depthFrame = qdepth.tryGet()
			imuData = imuQueue.tryGet()
			if imuData != None:
				if ROS and True:
					packets = imuData.packets
					for packet in packets:
						# FOR MOCKING:
						#packet = getMockIMU(packet)
						imu_msg = populate_imu(packet)
						imu_pub.publish(imu_msg)
					
			if depthFrame != None:
				depthFrame = depthFrame.getFrame()
				depthFrame+=1
				depthFrame =	numerator / depthFrame
				if FIRST < 0:
					rolling_frame[-1] = np.minimum(depthFrame,rolling_frame[-1])
				
				if not c % RATE and FIRST<0: # integrate over RATE frames
					markers = matrix_processing.process(rolling_frame[-1])
					flipped_frame = np.flip(rolling_frame[-1].T,axis=1)

					# set marker distances
					for i,elem in enumerate(markers):
						if int(elem) < CUTOFF:
							markers[i] = flipped_frame[i,int(elem)+50]
							flipped_frame[i,int(elem)+50:] = flipped_frame[i,int(elem)+50]
						else:
							markers[i] = numerator

					markers = np.flip(markers)
					prev_markers[0] = np.minimum(markers,prev_markers[0])
     
				if not c % RATE and FIRST<0:
					nval = matrix_processing.arr_conv(prev_markers[0])/9
					prev_markers[0] = nval
					# create point cloud
					x, dx = point_cloud(prev_markers[0])
					
					# Populate LaserScan message
					if ROS and True:
						laserscan_msg = populate_laserscan(x,dx)
						# Publish the LaserScan message
						laserscan_pub.publish(laserscan_msg)
					if DISPLAY:
						cv2.imshow("raw",cv2.hconcat([depthFrame/1000,np.flip(flipped_frame,axis=1).T/1000] ))
						cv2.imshow("scatter",scatterplot(rolling_frame[-1].shape ,x[:,np.newaxis]))
						cv2.waitKey(1)
					# reset
					rolling_frame = [np.zeros((400,640))+10000]
					prev_markers = [np.array([10000 for _ in range(640)])]
					c=0

				FIRST-=1
				if FIRST < 0:
					c+=1
				
a = 0
def getMockIMU(packet):
	global a
	# oops only X
	x = 0# X IS FRONT AND BACK
	y = 0
	z = 0
	real = 0
	zero = 0
	a+=1 # btw i think its a validation thing similar how it wasnt reading your previous data
	packet.acceleroMeter.x = 1	# linear velocity north/south
	packet.acceleroMeter.y = 0	# linear velocity east/west (when this is the only non zero number it does not look like its moving)
	packet.acceleroMeter.z = 0	# linear velocity up/down
	packet.rotationVector.i = 0	# unk when alone, combined with accel.x turning left/right? but so does j apparently?
	packet.rotationVector.j = 0 # unk
	packet.rotationVector.k = 0	# unk
	packet.rotationVector.real = 0
	packet.gyroscope.x = 0	 # unk
	packet.gyroscope.y = 0	# unk
	packet.gyroscope.z = 0	# paralell to floor, circular motion left/right # x=0, y=0, z=10 this makes it do point turns
	return packet

if __name__ == '__main__':
	main()
