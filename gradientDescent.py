import numpy as np 
import pandas as pd 
from costFunction import costFunction

def gradientDescent(X, y, theta, m, iteration, alpha):
	
	Jhistory = np.zeros((m, 1))
	for i in range(iteration):
		Hx = X.dot(theta)
		diff = np.subtract(Hx, y)
		Xtranspose = X.T 
		partialDerivative = (1.0/ float(m)) * (Xtranspose.dot(diff))
		A = theta - (alpha * partialDerivative)
		theta = A

		f = costFunction(X, y, theta, m)
		Jhistory = f 

	return theta