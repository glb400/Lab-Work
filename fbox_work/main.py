# main.py

import cv2
import foolbox
import keras
# from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import ResNet50, preprocess_input

from utils import *

# from adversarial_vision_challenge import load_model
# from adversarial_vision_challenge import read_images
# from adversarial_vision_challenge import store_adversarial
# from adversarial_vision_challenge import attack_complete

import numpy as numpy 
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys

# from resnet50 import *

sys.path.append('/home/wangguangrun/.local/lib/python3.5/site-packages')

def run_attack(model, image, label):
	# apply untargeted adversarial attack
	criterion = foolbox.criteria.Misclassification()
	# there're several attack method
	
	# the one based on gradient : 
	# FSGM (Fast Gradient Sign Method)
	# PGD (Project Gradient Descent)
	# MIM (Motentum Iterative Method)
	
	# the one based on optimize :
	# CW (Carlini-Wagner Attack)

	# other : DEEPFOOL / Pointwise

	# attack = foolbox.attacks.IterativeGradientAttack(model, criterion)
	# attack = foolbox.attacks.IterativeGradientSignAttack(model, criterion)
	attack = foolbox.attacks.CarliniWagnerL2Attack(model, criterion)
	adversarial = foolbox.Adversarial(model, criterion, image, label)
	attack(adversarial)
	print(adversarial.distance.value)

	# return attack(image, label)
	return adversarial.image

def main():

	#input data
	input_dir = '/home/wangguangrun/ILSVRC2012/val'
	Images, Labels = read_val_images(input_dir)

	print("Images.shape: ", Images.shape)

	# dtype float32, with values between 0 & 255
	# label is the original label (we use untargeted attacks)
	# adversarial = run_attack(model, image, label)
	# store_adversarial = (filename, adversarial)

	# instantiate model
	keras.backend.set_learning_phase(0)
	kmodel = ResNet50(weights='imagenet')
	fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255))

	#attack
	for idx in range(50000):
		# adversarial = run_attack(model=fmodel, image=Images[idx], label=Labels[idx])

		attack  = foolbox.attacks.FGSM(fmodel)
		adversarial = attack(Images[idx], Labels[idx])

		# x = adversarial.tobytes()
		# img = cv2.imdecode(np.fromstring(x, np.uint8), cv2.IMREAD_COLOR)
		# cv2.imwrite(str(idx) + '.jpeg', img)	
		# print(img)

		im = Image.fromarray(np.uint8(adversarial))
		im.save( str(idx) + ".jpeg")
	
	# attack_complete()	

