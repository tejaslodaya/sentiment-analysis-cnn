import numpy as np
def max_pool(C_i):
	max_val=np.max(C_i)
	max_index=np.argmax(C_i)

	return max_val,max_index