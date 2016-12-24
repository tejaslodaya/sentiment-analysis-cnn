import numpy as np
import pickle
import itertools
from max_pool import *
from softmax import *
from forward_propogation import *
from backpropogation import *

correct=0
num_classes=int(input("Enter the number of classes 3/5"))
Why=np.random.randn(num_classes,3)

Wxh=[np.random.uniform(-np.sqrt(1./32),np.sqrt(1./32),(64)),
np.random.uniform(-np.sqrt(1./32),np.sqrt(1./32),(96)),
np.random.uniform(-np.sqrt(1./32),np.sqrt(1./32),(128))]

dWxh=[np.zeros_like(x) for x in Wxh]

learning_rate=float(input("Enter the learning rate"))
epoch=int(input("Enter the number of epochs"))

train(num_classes)
test(num_classes)