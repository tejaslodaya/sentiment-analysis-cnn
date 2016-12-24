import numpy as np
import pickle
import itertools

def train(num_classes=5):

	sent=pickle.load(open("sent.pkl","rb"))
	labels=pickle.load(open("labels.pkl","rb"))

	sent=sent[:50000]

	for qw in range(epoch):
		for k in range(len(sent)):
			sentence=sent[k]
			class_pred,ps,C_values,c_list,C_index,lnump=forward(sentence,k,labels,num_classes)
		
			dWxh=backpropogation(ps,C_values,c_list,C_index,labels,k,lnump)

			for i in range(len(C_values)):
				Wxh[i] += learning_rate * dWxh[i]
		print("Epoch ",(qw+1)," done")
	print("Training done!")
