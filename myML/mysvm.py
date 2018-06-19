import numpy as np
import matplotlib.pyplot as plt

class MySVM:
    def __init__(self):

    def fit(self, x, y):
        
        

data = np.loadtxt("myteach.csv", delimiter= ",")
print(data)
row , col = data.shape

x = data[:, :col-1].T

a = np.ones((1,row))

x = np.vstack((x,a))

y = np.matrix(data[:, col-1]).T
y[y==0] = -1
print(x)
print(y)
y = np.array(y)

R = np.dot(x.T ,x)
A = np.zeros((row, row))
print("R",R)


for i in range(row):
    for j in range(row):
        A[i][j] = y[i]*y[j]*R[i][j]

print("A",A)

O = np.ones((row,1))
a = np.zeros((row,1))
dL = -1 * np.dot(A,a) + O
k=0
while(k<100):
    for l in range(row):
        
        a[l,0] = a[l,0] + 0.1 * dL[l,0]
        
        if a[l,0] < 0:
            a[l,0] = 0
            continue
        if a[l,0] > 3:
            a[l,0] = 3
            continue
    dL = -1 * np.dot(A,a) + O
    k += 1

print("a0=")
print(a)

psi = np.dot(x,a*y)
print("")
print("psi", psi)

psi = np.array(psi)
"""
test = np.loadtxt("mytest.csv", delimiter= ",")

test = np.hstack((test, np.ones((len(test),1))))
print(test)

for i in test:
    judge = np.dot(i,psi)
    if judge < 0:
        print("spam")
    elif judge > 0:
        print("ham")
    else:
        pass


"""

x_fig = np.array(range(-3,3))
y_fig = -psi[0,0]/psi[1,0] * x_fig - psi[2,0]/psi[1,0]

plt.scatter(x[0,0:row//2],x[1,0:row//2],c="blue")

plt.scatter(x[0,row//2:],x[1,row//2:],c="red")
plt.xlabel("x")
plt.ylabel("y")

plt.plot(x_fig, y_fig)
plt.show()
