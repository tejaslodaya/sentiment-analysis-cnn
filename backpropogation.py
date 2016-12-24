import numpy as np
import pickle
import itertools

def backpropogation(ps,C_values,c_list,C_index,labels,k,lnump):
	######################BACKPROPOGATION#############
	dWhy=np.zeros_like(Why)

	dy=np.copy(ps)
	dy[labels[k]] -= 1

	dy=np.reshape(dy,(len(dy),1))
	C_values=np.reshape(C_values,(len(C_values),1))

	dWhy += np.dot(dy,C_values.T)

	dc=np.dot(Why.T,dy) #3*1

	for i in range(len(C_values)):
		dWxh[i]=np.zeros_like(Wxh[i])
		max_index=C_index[i]

		dhraw=(1-c_list[i][max_index]**2) * dc[i]
		dhraw=np.reshape(dhraw,(len(dhraw),1))

		temp=lnump[i][max_index].T
		temp=np.reshape(temp,(1,len(temp)))

		cool=np.dot(dhraw,temp)
		cool=np.reshape(cool,(len(cool[0])))

		dWxh[i]=cool

	return dWxh;