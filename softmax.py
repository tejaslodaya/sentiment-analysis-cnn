import numpy as np
def softmax(x):
    exp_scores=np.exp(x)
    probs=exp_scores/np.sum(exp_scores)
    return probs