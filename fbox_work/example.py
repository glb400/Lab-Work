import foolbox
import keras
from keras.applications.resnet50 import ResNet50, preprocess_input

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# instantiate model
keras.backend.set_learning_phase(0)
kmodel = ResNet50(weights='imagenet')
fmodel = foolbox.models.KerasModel(kmodel, bounds=(0, 255),preprocess_fn=preprocess_input)

# get source image and label
image, label = foolbox.utils.imagenet_example()

# apply attack on source image
attack  = foolbox.attacks.FGSM(fmodel)
adversarial = attack(image, label)