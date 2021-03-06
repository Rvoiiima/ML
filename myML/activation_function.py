import numpy as np
def step_function(x):
	y = x > 0
	return y.astype(np.int)

def sigmoid(x):
	return 1 /(1 + np.exp(-x))

def ReLU(x):
	return np.maximum(0, x)

def softmax(x):
	c = np.max(x)
	exp_x = np.exp(x-c)
	sum_exp_x = np.sum(exp_x)
	y = exp_x / sum_exp_x

	return y

def main():
	x = np.array([1,-2,3])

	print(sigmoid(x))
	print(ReLU(x))

if __name__ == "__main__":
	main()
