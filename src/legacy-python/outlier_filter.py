#!/usr/bin/python3
import numpy as np

def get_1d_gaussian(filter_size=3, sigma=1):
	x = np.linspace(-filter_size / 2, filter_size/2, filter_size)
	gaussian_filter_1d = np.exp(-x**2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)
	return gaussian_filter_1d
	
def get_mean_filter(filter_size = 9):
	return np.ones((filter_size, 1)) / filter_size

def get_vec_filter(f):
	return f[:,np.newaxis]

def normalize(arx):
	#arx = array * (6618.75 - 1)
	#arx = (arx+1) / (7.5 * 882.5) * 95 + 1
	#arx = (7.5 * 882.5)/arx
	val = np.max(np.max(arx))
	oval = np.min(np.min(arx))
	arx = (arx - oval) / (val - oval) * -1 + 1
	arx = arx * (val - oval)
	return arx

def outlier_rejection(mean_deriv):
	"""
	Vectorize IQR 
	"""
	# Calculate quartiles
	Q1 = np.percentile(mean_deriv[:,:], 25,axis=1)
	Q3 = np.percentile(mean_deriv[:,:],75, axis=1)

	# Calculate IQR
	IQR = Q3 - Q1
	threshold = 1.5 * IQR[:]
	lb = Q1 - threshold
	ub = Q3 + threshold

	#for row in range(mean_deriv.shape[0]):
	#	mean_deriv[row,:] = lb[row] if mean_deriv[row,:] < lb[row] else mean_deriv[row,:]
	#	mean_deriv[row,:] = ub[row] if mean_deriv[row,:] > ub[row] else mean_deriv[row,:]
		#for col in range(mean_deriv.shape[1]):
			#mean_deriv[row,col] = lb[row] if mean_deriv[row,col] < lb[row] else mean_deriv[row,col]
			#mean_deriv[row,col] = ub[row] if mean_deriv[row,col] > ub[row] else mean_deriv[row,col]
	
	return np.clip(mean_deriv, lb[:,np.newaxis], ub[:,np.newaxis])

# the filters
gauss_f = get_vec_filter(get_1d_gaussian(5))
gauss_delay = gauss_f.shape[0]
mean_f = get_vec_filter(get_mean_filter(9))
mean_delay = mean_f.shape[0]
deriv_f = get_vec_filter(np.array([-1,1]))

def find_obs(frame,lower_bound = 50, upper_bound = 350):
	"""
	Image processing
	"""
	conv = lambda x,h: np.apply_along_axis(lambda x: np.convolve(x, h.flatten(), mode='full'),axis=1,arr=x)
	
	# normalize and invert
	frame = normalize(frame)
	
	# blur column by applying gaussian filter
	blurred_frame = conv(frame, gauss_f)

	# apply window to blurred frame
	bounded_frame = blurred_frame[:,lower_bound:upper_bound]
	
	# apply mean filter
	mean_filtered_frame = conv(bounded_frame, mean_f)
	
	# get first derivative
	first_derivative = conv(mean_filtered_frame[:,:],deriv_f)
	
	# outlier rejection
	first_derivative = outlier_rejection(first_derivative)
	
	# initialize obstacle matrix to all free spaces
	obs_matrix = np.ones((frame.shape[0],upper_bound - lower_bound))
		
	# threshold with mean of each row
	mean_arr = np.mean(first_derivative[:,:],axis=1)
	
	filter_padding = mean_delay + gauss_delay
	for i in range(first_derivative.shape[0]):
		# trim convolution heads and tails, shift obstacles by filter sizes
		for j in range(filter_padding,first_derivative.shape[1]-filter_padding):
			if np.sign(first_derivative[i,j]) != np.sign(first_derivative[i,j-1]) or first_derivative[i,j] < mean_arr[i]/4:
				obs_matrix[i,j:-1] = 0
				break
					
	return obs_matrix

def get_free_space(rect_arr, frame):
	"""
	Vectorized implementation which extends present obstacles to the frame boundary
	"""
	free_spaces = frame
	posns = np.argmax(rect_arr == 0, axis=1)
	posns[posns == 0] = rect_arr.shape[1]
	for i,p in enumerate(posns):
		if p != rect_arr.shape[1]:
			free_spaces[i,50 + p:-1] = frame[i,50 + p -1]
	return free_spaces

def get_markers(rect_arr):
	"""
	Get obstacle in every column if it exists
	"""
	markers = np.argmin(rect_arr[:,:] == 1,axis=1)
	return markers

def _get_free_space(rect_arr, arm):
	free_spaces = np.zeros(arm.shape)
	for i in range(arm.shape[0]):
		permj = 0
		for j in range(arm.shape[1]):
			if j < 50:
				free_spaces[i,j] = arm[i,j]
			elif j > 349:
				free_spaces[i,j] = free_spaces[i,j-1]
			else:
				if rect_arr[i,j-50] != 0 and permj == 0:
					free_spaces[i,j] = arm[i,j]
				else:
					if permj == 0:
						permj = j - 1
					free_spaces[i,j:-1] = arm[i,permj]
					break
	return free_spaces


				
			
	
	
	
	
	
	
	
	
	
	
	
	
