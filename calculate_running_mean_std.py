from tqdm import tqdm
from PIL import Image
import numpy as np
import json

for some_file in ['/mnt/c/Users/vlade813/Desktop/School/Masters/Spring 2023/Deep Learning/CSD-SSD-COMET/voc2007.json',
				  '/mnt/c/Users/vlade813/Desktop/School/Masters/Spring 2023/Deep Learning/CSD-SSD-COMET/voc2012.json']:
	# some_file = '/mnt/c/Users/vlade813/Desktop/School/Masters/Spring 2023/Deep Learning/CSD-SSD-COMET/voc0712.json'
	f = json.load(open(some_file,'r'))

	x_mean = np.asarray(0,dtype=np.float64)
	x_var  = np.asarray(0,dtype=np.float64)
	x_pix  = np.asarray(0,dtype=np.float64)

	im_list = []
	num_imgs = len(f['images'])
	for i, im_dict in enumerate(tqdm(f['images'])):
		im = Image.open(im_dict['file_name'])
		im = np.asarray(im)
		im = im.reshape((im.shape[0]*im.shape[1],3))

		im_list.append(im)

		j = i+1
		if (j % 25 == 0) or (j == num_imgs):

			im = np.concatenate(im_list,axis=0)
			# print(im.shape)

			# used formulas from the following: https://math.stackexchange.com/questions/3604607/can-i-work-out-the-variance-in-batches
			num_pixels = np.asarray(im.shape[0],dtype=np.float64)
			im_mean = np.sum(im, axis=0, dtype=np.float64)/num_pixels
			im_var  = np.sum(np.square(im, dtype=np.float64), axis=0, dtype=np.float64)/num_pixels - im_mean**2

			# calculate total variance
			left  = ((num_pixels-1)*im_var + (x_pix-1)*x_var)/(x_pix + num_pixels - 1)
			right = (num_pixels*x_pix*(x_mean - im_mean)**2)/((x_pix + num_pixels - 1)*(x_pix + num_pixels))
			x_var = left + right

			# calculate the total mean
			x_mean = (x_mean*x_pix + im_mean*num_pixels) / (x_pix + num_pixels)
			x_pix += num_pixels

			im_list = []

		# if j % 5000 == 0:
		# 	print("---------------")
		# 	print(f"idx = {i}")
		# 	print("mean: ",x_mean)
		# 	print("var:  ",x_var)
		# 	print("---------------")

	mean = x_mean
	std  = np.sqrt(x_var)

	print("---------------")
	print("Dataset mean: ",mean)
	print("Dataset std:  ",std)
	print("---------------")
