import os
import cv2
import itertools
import math
from typing import Tuple
import numpy as np
from scipy.interpolate import interp1d
import scipy.io as sio
import matplotlib.pyplot as plt


# Sub-Functions
#######################################################################################################################################################################

# implementazione algoritmo di daugman per la detection dell'iride
def daugman(img, start_r, center):

	# get separate coordinates
	x, y = center
	
	# get img dimensions
	h, w = img.shape

	# define some other vars
	tmp = []
	mask = np.zeros_like(img)

	# k è un parametro per determinare il limite superiore dei raggi
	k = h
	if w < h:
		k = w

	# for every radius in range
	# we are presuming that iris will be no bigger than 1/3 of picture
	for r in range(start_r, int(k / 3)):
		
		# draw circle on mask (solo la circonferenza non cerchio pieno!)
		cv2.circle(mask, center, r, 255, 1)

		# get pixel from original image
		radii = img & mask  # it is faster than np or cv2

		# normalize np.add.reduce faster than .sum()
		tmp.append(np.add.reduce(radii[radii > 0]) / (2 * math.pi * r))
		
		# refresh mask
		mask.fill(0)

	# calculate delta of radius intensitiveness
	# mypy does not tolerate var type reload
	tmp_np = np.array(tmp, dtype=np.float32)
	del tmp

	if tmp_np != []:
		tmp_np = tmp_np[1:] - tmp_np[:-1]  # x5 faster than np.diff()

		# aply gaussian filter
		tmp_np = abs(cv2.GaussianBlur(tmp_np[:-1], (1, 5), 0))

		# get index of the maximum value
		idx = np.argmax(tmp_np)

		# get maximum value
		val = tmp_np[idx]

		circle = (center, idx + start_r)
	else:
		val = 0
		circle = ((0, 0), 0)

	# return value, center coords, radius
	return val, circle

def find_iris(img, start_r):
	values = []
	coords = []

	h, w = img.shape

	for i in range(0 + int(h / 3), h - int(h / 3), 3):
		for j in range(0 + int(w / 3), w - int(w / 3), 3):
			tmp = daugman(img, start_r, (j, i))
			if tmp is not None:
				val, circle = tmp
				values.append(val)
				coords.append(circle)
	
	# return the radius with biggest intensiveness delta on image
	# ((xc, yc), radius)
	return coords[values.index(max(values))]


# trovo la pupilla tramite l'algoritmo di daugman
def pupil_detection(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	res = find_iris(gray, 10)
	return  res[0][0], res[0][1], res[1]

# trovo la sclera tramite l'algoritmo di daugman
def iris_contour_detection(img, pupil_radius, alpha):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	res = find_iris(gray, pupil_radius*alpha)
	return  res[0][0], res[0][1], res[1]


# creo la maschera per segmentare l'iride
def create_circular_mask(shape, circle):
	(x, y, r) = circle
	mask = np.zeros(shape, dtype=np.uint8)
	cv2.circle(mask, (x, y), r, (255, 255, 255), -1, 8, 0)
	return mask


# normalizzo l'iride segmentata in coordinate polari tramite il dougman rubber sheet model
def iris_normalization(image, x_iris, y_iris, r_iris, x_pupil, y_pupil, r_pupil, radpixels, angulardiv):
	"""
	Description:
		Normalize iris region by unwraping the circular region into a rectangular
		block of constant dimensions.
	Input:
		image		- Input iris image.
		x_iris		- x-coordinate of the circle defining the iris boundary.
		y_iris		- y-coordinate of the circle defining the iris boundary.
		r_iris		- Radius of the circle defining the iris boundary.
		x_pupil		- x-coordinate of the circle defining the pupil boundary.
		y_pupil		- y-coordinate of the circle defining the pupil boundary.
		r_pupil		- Radius of the circle defining the pupil boundary.
		radpixels	- Radial resolution (vertical dimension).
		angulardiv	- Angular resolution (horizontal dimension).
	Output:
		polar_array	- Normalized form of the iris region.
		polar_noise	- Normalized form of the noise region.
	"""
	radiuspixels = radpixels + 2
	angledivisions = angulardiv-1

	r = np.arange(radiuspixels)
	theta = np.linspace(0, 2*np.pi, angledivisions+1)

	# Calculate displacement of pupil center from the iris center
	ox = x_pupil - x_iris
	oy = y_pupil - y_iris

	if ox <= 0:
		sgn = -1
	elif ox > 0:
		sgn = 1

	if ox==0 and oy > 0:
		sgn = 1

	a = np.ones(angledivisions+1) * (ox**2 + oy**2)

	# Need to do something for ox = 0
	if ox == 0:
		phi = np.pi/2
	else:
		phi = np.arctan(oy/ox)

	b = sgn * np.cos(np.pi - phi - theta)

	# Calculate radius around the iris as a function of the angle
	r = np.sqrt(a)*b + np.sqrt(a*b**2 - (a - r_iris**2))
	r = np.array([r - r_pupil])

	rmat = np.dot(np.ones([radiuspixels,1]), r)

	rmat = rmat * np.dot(np.ones([angledivisions+1,1]),
							np.array([np.linspace(0,1,radiuspixels)])).transpose()
	rmat = rmat + r_pupil

	# Exclude values at the boundary of the pupil iris border, and the iris scelra border
	# as these may not correspond to areas in the iris region and will introduce noise.
	# ie don't take the outside rings as iris data.
	rmat = rmat[1 : radiuspixels-1, :]

	# Calculate cartesian location of each data point around the circular iris region
	xcosmat = np.dot(np.ones([radiuspixels-2,1]), np.array([np.cos(theta)]))
	xsinmat = np.dot(np.ones([radiuspixels-2,1]), np.array([np.sin(theta)]))

	xo = rmat * xcosmat
	yo = rmat * xsinmat

	xo = x_pupil + xo
	xo = np.round(xo).astype(int)
	coords = np.where(xo >= image.shape[1])
	xo[coords] = image.shape[1] - 1
	coords = np.where(xo < 0)
	xo[coords] = 0
	
	yo = y_pupil - yo
	yo = np.round(yo).astype(int)
	coords = np.where(yo >= image.shape[0])
	yo[coords] = image.shape[0] - 1
	coords = np.where(yo < 0)
	yo[coords] = 0

	# Extract intensity values into the normalised polar representation through
	# interpolation
	# x,y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
	# f = interpolate.interp2d(x, y, image, kind='linear')
	# polar_array = f(xo, yo)
	# polar_array = polar_array / 255

	polar_array = image[yo, xo]
	polar_array = polar_array / 255

	# Create noise array with location of NaNs in polar_array
	polar_noise = np.zeros(polar_array.shape)
	coords = np.where(np.isnan(polar_array))
	polar_noise[coords] = 1

	# Get rid of outling points in order to write out the circular pattern
	image[yo, xo] = 255

	# Get pixel coords for circle around iris
	x,y = circlecoords([x_iris,y_iris], r_iris, image.shape)
	image[y,x] = 255

	# Get pixel coords for circle around pupil
	xp,yp = circlecoords([x_pupil,y_pupil], r_pupil, image.shape)
	image[yp,xp] = 255

	# Replace NaNs before performing feature encoding
	coords = np.where((np.isnan(polar_array)))
	polar_array2 = polar_array
	polar_array2[coords] = 0.5
	avg = np.sum(polar_array2) / (polar_array.shape[0] * polar_array.shape[1])
	polar_array[coords] = avg

	return polar_array, polar_noise.astype(bool)

def circlecoords(c, r, imgsize, nsides=600):
	"""
	Description:
		Find the coordinates of a circle based on its centre and radius.
	Input:
		c   	- Centre of the circle.
		r  		- Radius of the circle.
		imgsize - Size of the image that the circle will be plotted onto.
		nsides 	- Number of sides of the convex-hull bodering the circle
				  (default as 600).
	Output:
		x,y     - Circle coordinates.
	"""
	a = np.linspace(0, 2*np.pi, 2*nsides+1)
	xd = np.round(r * np.cos(a) + c[0])
	yd = np.round(r * np.sin(a) + c[1])

	#  Get rid of values larger than image
	xd2 = xd
	coords = np.where(xd >= imgsize[1])
	xd2[coords[0]] = imgsize[1] - 1
	coords = np.where(xd < 0)
	xd2[coords[0]] = 0

	yd2 = yd
	coords = np.where(yd >= imgsize[0])
	yd2[coords[0]] = imgsize[0] - 1
	coords = np.where(yd < 0)
	yd2[coords[0]] = 0

	x = np.round(xd2).astype(int)
	y = np.round(yd2).astype(int)
	return x,y


# estrazione features tramite gabor wavelets
def gaborconvolve(im, minWaveLength, mult, sigmaOnf):
	"""
	Description:
		Convolve each row of an image with 1D log-Gabor filters.
	Input:
		im   			- The image to be convolved.
		minWaveLength   - Wavelength of the basis filter.
		mult   			- Multiplicative factor between each filter.
		sigmaOnf   		- Ratio of the standard deviation of the
						  Gaussian describing the log Gabor filter's transfer
						  function in the frequency domain to the filter center
						  frequency.
	Output:
		filterbank		- The 1D cell array of complex valued convolution
						  resultsCircle coordinates.
	"""
	# Pre-allocate
	rows, ndata = im.shape					# Size
	logGabor = np.zeros(ndata)				# Log-Gabor
	filterbank = np.zeros([rows, ndata], dtype=complex)

	# Frequency values 0 - 0.5
	radius = np.arange(ndata/2 + 1) / (ndata/2) / 2
	radius[0] = 1

	# Initialize filter wavelength
	wavelength = minWaveLength

	# Calculate the radial filter component
	fo = 1 / wavelength 		# Centre frequency of filter
	logGabor[0 : int(ndata/2) + 1] = np.exp((-(np.log(radius/fo))**2) / (2 * np.log(sigmaOnf)**2))
	logGabor[0] = 0

	# For each row of the input image, do the convolution
	for r in range(rows):
		signal = im[r, 0:ndata]
		imagefft = np.fft.fft(signal)
		filterbank[r , :] = np.fft.ifft(imagefft * logGabor)

	# Return
	return filterbank

# generazione codice tramite quantizzazione della fase
def encode(polar_array, noise_array, minWaveLength, mult, sigmaOnf):
	"""
	Description:
		Generate iris template and noise mask from the normalised iris region.
	Input:
		polar_array		- Normalised iris region.
		noise_array		- Normalised noise region.
		minWaveLength	- Base wavelength.
		mult			- Multicative factor between each filter.
		sigmaOnf		- Bandwidth parameter.
	Output:
		template		- The binary iris biometric template.
		mask			- The binary iris noise mask.
	"""
	# Convolve normalised region with Gabor filters
	filterbank = gaborconvolve(polar_array, minWaveLength, mult, sigmaOnf)

	length = polar_array.shape[1]
	template = np.zeros([polar_array.shape[0], 2 * length])
	h = np.arange(polar_array.shape[0])

	# Create the iris template
	mask = np.zeros(template.shape)
	eleFilt = filterbank[:, :]

	# Phase quantization
	H1 = np.real(eleFilt) > 0
	H2 = np.imag(eleFilt) > 0

	# If amplitude is close to zero then phase data is not useful,
	# so mark off in the noise mask
	H3 = np.abs(eleFilt) < 0.0001
	for i in range(length):
		ja = 2 * i

		# Construct the biometric template
		template[:, ja] = H1[:, i]
		template[:, ja + 1] = H2[:, i]

		# Create noise mask
		mask[:, ja] = noise_array[:, i] | H3[:, i]
		mask[:, ja + 1] = noise_array[:, i] | H3[:, i]

	# Return
	return template, mask

#######################################################################################################################################################################



# Main Functions
#######################################################################################################################################################################

def iris_encoding(img):

	# reduce/remove noise applicando filtro gaussiano
	blur = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)

	# contrast enhancement
	normalized = cv2.normalize(blur, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F) # from 0-255 -> 0-1
	gamma = 1.2
	contrasted = np.power(normalized, 1/gamma)
	contrasted = cv2.normalize(contrasted, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1) # from 0-1 -> 0-255

	output = img.copy()

	# detection
	(x1, y1, r1) = pupil_detection(contrasted)
	(x2, y2, r2) = iris_contour_detection(contrasted, r1, 2)

	#cv2.circle(output, (x1, y1), r1, (0, 255), 2)
	#cv2.circle(output, (x2, y2), r2, (0, 255), 2)

	#print('centro pupilla -> x: {}, y: {}'.format(x1, y1))
	#print('centro iride -> x: {}, y: {}'.format(x2, y2))

	#plt.imshow(output)
	#plt.title('iris detection')
	#plt.show()

	# masking
	pupil_mask = create_circular_mask(img.shape[:2], (x1, y1, r1))
	external_iris_mask = create_circular_mask(img.shape[:2], (x2, y2, r2))
	mask = np.subtract(external_iris_mask, pupil_mask)

	#plt.imshow(mask, cmap='gray')
	#plt.title('mask')
	#plt.show()


	# isolated iris
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	isolated_iris = gray & mask

	#plt.imshow(isolated_iris, cmap='gray')
	#plt.title('isolated_iris')
	#plt.show()


	# iris normalization
	polar_array, noise_array = iris_normalization(isolated_iris, x2, y2, r2, x1, y1, r1, 20, 240)

	#plt.imshow(polar_array, cmap='gray')
	#plt.title('polar_array')
	#plt.show()


	# feature extraction (2D gabor wavelet)
	# generate code (phase quantization)
	minWaveLength = 18
	mult = 1
	sigmaOnf = 0.5
	template, mask = encode(polar_array, noise_array, minWaveLength, mult, sigmaOnf)

	#plt.imshow(template, cmap='gray')
	#plt.title('template')
	#plt.show()

	return template, mask

def dataset_encoding(data_path):
	subjects = [filename for filename in os.listdir(data_path) if filename != "ReadMe.txt"]
	subjects.sort()
	for subject in subjects:
		leftpath = data_path + '/' + subject + '/left'
		left = [filename for filename in os.listdir(leftpath) if filename.split('.')[1] == 'bmp']

		for x in left:
			image = cv2.imread(leftpath + '/' + x)
			template, mask = iris_encoding(image)
			filepath = leftpath + '/' + x.split('.')[0] + '.mat'
			sio.savemat(filepath, mdict={'template': template, 'mask': mask, 'name': x.split('.')[0]})

		rightpath = data_path + '/' + subject + '/right'
		right = [filename for filename in os.listdir(rightpath) if filename.split('.')[1] == 'bmp']
		for x in right:
			image = cv2.imread(rightpath + '/' + x)
			template, mask = iris_encoding(image)
			filepath = rightpath + '/' + x.split('.')[0] + '.mat'
			sio.savemat(filepath, mdict={'template': template, 'mask': mask, 'name': x.split('.')[0]})

		print("subject: {} encoded!".format(subject))

def hammingdist(template1, mask1, template2, mask2):
	"""
	Description:
		Calculate the Hamming distance between two iris templates.
	Input:
		template1	- The first template.
		mask1		- The first noise mask.
		template2	- The second template.
		mask2		- The second noise mask.
	Output:
		hd			- The Hamming distance as a ratio.
	"""
	# Initialize
	hd = np.nan

	# Shift template left and right, use the lowest Hamming distance
	for shifts in range(-8,9):
		template1s = shiftbits(template1, shifts)
		mask1s = shiftbits(mask1, shifts)

		mask = np.logical_or(mask1s, mask2)
		nummaskbits = np.sum(mask==1)
		totalbits = template1s.size - nummaskbits

		C = np.logical_xor(template1s, template2)
		C = np.logical_and(C, np.logical_not(mask))
		bitsdiff = np.sum(C==1)

		if totalbits==0:
			hd = np.nan
		else:
			hd1 = bitsdiff / totalbits
			if hd1 < hd or np.isnan(hd):
				hd = hd1

	# Return
	return hd

def shiftbits(template, noshifts):
	"""
	Description:
		Shift the bit-wise iris patterns.
	Input:
		template	- The template to be shifted.
		noshifts	- The number of shift operators, positive for right
					  direction and negative for left direction.
	Output:
		templatenew	- The shifted template.
	"""
	# Initialize
	templatenew = np.zeros(template.shape)
	width = template.shape[1]
	s = 2 * np.abs(noshifts)
	p = width - s

	# Shift
	if noshifts == 0:
		templatenew = template

	elif noshifts < 0:
		x = np.arange(p)
		templatenew[:, x] = template[:, s + x]
		x = np.arange(p, width)
		templatenew[:, x] = template[:, x - p]

	else:
		x = np.arange(s, width)
		templatenew[:, x] = template[:, x - s]
		x = np.arange(s)
		templatenew[:, x] = template[:, p + x]

	# Return
	return templatenew

def recognition(template1, mask1, data_path):
	result = False
	values = []
	
	threshold = 0.38

	subjects = [filename for filename in os.listdir(data_path) if filename != "ReadMe.txt"]
	subjects.sort()
	
	for subject in subjects:

		leftpath = data_path + '/' + subject + '/left'
		left = [filename for filename in os.listdir(leftpath) if filename.split('.')[1] == 'bmp']
		for x in left:
			matfile = sio.loadmat(leftpath + '/' + x.split('.')[0] + '.mat')
			template2 = matfile['template']
			mask2 = matfile['mask']
			result = hammingdist(template1, mask1, template2, mask2)

			values.append(result)
			
			if result <= threshold:
				print('soggetto: {}, result: {}'.format(leftpath + '/' + x.split('.')[0], result))

		rightpath = data_path + '/' + subject + '/right'
		right = [filename for filename in os.listdir(rightpath) if filename.split('.')[1] == 'bmp']
		for x in right:
			matfile = sio.loadmat(rightpath + '/' + x.split('.')[0] + '.mat')
			template2 = matfile['template']
			mask2 = matfile['mask']
			result = hammingdist(template1, mask1, template2, mask2)

			values.append(result)
			
			if result <= threshold:
				print('soggetto: {}, result: {}'.format(rightpath + '/' + x.split('.')[0], result))

	return min(values)[0] <= threshold

#######################################################################################################################################################################



# Main Program
#######################################################################################################################################################################

os.environ['DISPLAY'] = ':0'
data_path = './data'

#dataset_encoding(data_path) # creazione db


choice = 0
filename = '/31/left/roslil1.bmp'

print('test soggetto: {}'.format(filename))

img = cv2.imread(data_path + filename)
template1, mask1 = iris_encoding(img)


if choice == 0:
	# test su tutto il db
	result = recognition(template1, mask1, data_path)
else:
	# test singolo
	filename2 = '/2/left/bryanl1.mat'

	print('confronto con il soggetto: {}'.format(filename2))

	matfile = sio.loadmat(data_path + '/' + filename2)
	template2 = matfile['template']
	mask2 = matfile['mask']
	result = hammingdist(template1, mask1, template2, mask2)

print("recognition result: {}".format(result))


#######################################################################################################################################################################