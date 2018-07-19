import matplotlib.pyplot as plt

def plotData(a, b):
	plt.scatter(a, b, marker="x")
	plt.xlabel('population of city in 10,000')
	plt.ylabel('profit in $10,000')
	plt.title('population vs profit')
	plt.show()