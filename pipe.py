##################################################################################### 
#  This is an implimentation of linear regression									#
#  The data contains city's population and profit on that population				#
#  																					#
#  In this project following things are needed:										#
#																					#
#		plotData.py																	#
#		costFunction.py																#
#		gradientDescent.py															#
#																					#
#####################################################################################

import numpy as np
import pandas as pd
from plotData import plotData 
from costFunction import costFunction
from gradientDescent import gradientDescent
#load the exampledata file
data = pd.read_csv('exampleData.txt')

#defining X and y
X = data.iloc[:, 0]
y = data.iloc[:, 1]
#no. of example m
m = len(y)

plotData(X, y)

#####################################################################################
#  we have single feature vector x and label vector y							    #	
#  from the plot we get the X vs y relationship is linear							#
#  here the hypothesis is h(x) = theta[0]X[0] + theta[1]X[1]						#
#####################################################################################    

a = np.ones((len(y), 1))
X = pd.DataFrame(X)
X.insert(loc=0, column='b', value=a)
y = pd.DataFrame(y)

theta = np.zeros((2,1))

J = costFunction(X, y, theta, m)
print "costFunction for initial values of theta is %f" %(J)

thetaCheck = np.array([[-1],[2]])
i = costFunction(X, y, thetaCheck, m)
print "costFunction for thetaCheck is %f" %(i)

#setting hyperparameter alpha and no, of iteration 

iteration = 1500
alpha = 0.01

#finding optimum theta 
theta = gradientDescent(X, y, theta, m, iteration, alpha)

print "the optimum theta is " 
print theta 
