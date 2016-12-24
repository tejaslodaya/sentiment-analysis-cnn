import numpy as np
import pickle
import itertools

def test(num_classes=5):
	sent=pickle.load(open("sent_test.pkl","rb"))
	labels=pickle.load(open("labels_test.pkl","rb"))

	sent=sent[:5000]
	for k in range(len(sent)):
		sentence=sent[k]
		class_pred,ps,C_values,c_list,C_index,lnump=forward(sentence,k,labels,num_classes)

		if(class_pred==labels[k]):
			correct+=1
	print("Accuracy : ",correct/len(sent))
