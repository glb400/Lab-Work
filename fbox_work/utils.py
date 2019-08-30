# utils.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import numpy as np
import h5py
import sys

sys.path.append('/home/wangguangrun/.local/lib/python3.5/site-packages')

import foolbox
import copy
import os
import h5py
from scipy import misc
import imageio

from PIL import Image

import piexif

IMG_EXTENSIONS = ['.JPEG', '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def ClassMapping(map_file):
	class_map = dict()
	class_id = 0
	with open(map_file) as fin:
		for line in fin:
			list_t = line
			list_key = list_t[0:9]
			class_map[list_key] = int(class_id)
			class_id = class_id + 1
	return class_map

def read_val_images(input_dir):
	# data_list - save image
	data_list = list()
	# label_list - save label
	label_list = list()
	# class_map - map from class name(nXXXXXXXX) to value (0-999)
	class_map = ClassMapping("./wnids.txt")

	num_images = 0
	# num_test = 0

	# load image from tiny-ImageNet
	for key in class_map.keys():
		num_images = 0
		# num_test += 1
		
		# load original image
		path = input_dir + "/" + key
		for fi in os.listdir(path):
			if not (any(fi.endswith(ext) for ext in IMG_EXTENSIONS)):
				continue

			try:
				img = Image.open(path + "/" + fi)
				print(path + "/" + fi)
				
				img = img.convert("RGB")

				height, width = img.size[:2]
				print(img.size)
				
				if height > width:
					img = img.resize((int(height * 256 / width),256),Image.ANTIALIAS)
				else: 
					img = img.resize((256, int(width * 256 / height)),Image.ANTIALIAS)
				print(img.size)

				height, width = img.size[:2]
				lef = int(width / 2 - 112)
				upp = int(height / 2 - 112)
				rig = int(width / 2 + 112)
				low = int(height / 2 + 112)
				img = img.crop((lef, upp, rig, low))
				print(img.size)

				# img = img.resize((64,64),Image.ANTIALIAS)

			except IOError:
				print(path + "/" + fi)

			try:
				img = np.array(img, dtype=np.float32)
				print('bingo')
			except:
				print('corrupt img', path + "/" + fi)
				
			data_list.append(img)
			label_list.append(class_map[key])

#			print(class_map[key])

			num_images += 1
			if num_images > 50:
				break
		# if num_test >= 1:
			# break

	return np.array(data_list, dtype=np.float32), np.array(label_list, dtype=np.uint8)

if __name__ == "__main__":
	print("Test")
