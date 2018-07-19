import numpy as np 
import pandas as pd 

def costFunction(X, y, theta, m):

	#hypothesis
	Hx = X.dot(theta)
	#calculating mean square error
	diff = np.subtract(Hx, y)
	squareError = np.square(diff)
	totalError = np.sum(squareError)
	J = (1.0/float(2 * m)) * totalError

	return J