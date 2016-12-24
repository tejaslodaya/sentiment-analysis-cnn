import numpy as np
import pickle
import itertools

def forward(sentence,k,labels,num_classes=5):

	length=len(sentence)
	if length<4:
		diff=4-length
		for i in range(diff):
			sentence.append(np.asarray([0.0 for xy in range(32)]))

	lnump=[]
	l=[]
	for i in range(2,5):
	    l2nump=[]
	    l2=[]
	    for j in range(len(sentence)-i+1):
	        s=[]
	        for x in range(j,j+i):
	            s.append(sentence[x])
	        l2nump.append(np.asarray(list(itertools.chain(*s))))
	        l2.append(list(itertools.chain(*s)))
	    lnump.append(np.asarray(l2nump))
	    l.append(l2)
	lnump=np.asarray(lnump)

	C_values=[0 for x in range(len(lnump))] #3 univariate vectors combined
	C_index=[0 for x in range(len(lnump))]  #Retain indexes for backprop

	c_list=[]

	for i in range(len(lnump)): #Each gram
		
		C_i=np.empty(len(lnump[i]))

		for j in range(len(lnump[i])):
			cur_lnump=lnump[i][j]
			C_i[j]=np.tanh(Wxh[i].T.dot(cur_lnump))

		c_list.append(C_i)
		max_C,max_index=max_pool(C_i)
		C_values[i]=max_C
		C_index[i]=max_index

	c_list=np.asarray(c_list)
	C_values=np.asarray(C_values)

	p=np.dot(Why,C_values)	
	ps=softmax(p)
	class_pred=np.argmax(ps)

	return class_pred,ps,C_values,c_list,C_index,lnump