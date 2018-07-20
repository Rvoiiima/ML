import numpy as np
from activation_function import *

def init_network():
	W0 = np.array([[0.1, 0.3, 0.5],
					[0.2, 0.4, 0.6],
					[0.1, 0.2, 0.3]])

	W1 = np.array([[0.1, 0.4],
					[0.2, 0.5],
					[0.3, 0.6],
					[0.1, 0.2]])


	W2 = np.array([[0.1, 0.3],
					[0.2, 0.4],
					[0.1, 0.2]])

	W = [W0, W1, W2]

	return W


def forward(W, x):
	x.append(1)
	Z = [x]

	for i in range(2):
	
		Z.append(sigmoid(np.dot(Z[i], W[i])))
		Z[i+1] = np.append(Z[i+1], 1)

	
	y = softmax(np.dot(Z[2], W[2]))

	return y
	

def main():
	W = init_network()
	x = [1.0, 0.5]

	y = forward(W, x)

	print(y)


if __name__ == "__main__":
	main()
