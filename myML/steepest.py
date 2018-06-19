import numpy as np
A1 = np.array([[6,1],[1,2]])
b1 = np.array([[2],[1]])

def f(A,b,x):
    
    return 0.5*np.dot(np.dot(x.T,A),x)-np.dot(b.T,x)

def df(A,b,x):
    return np.dot(A,x)-b

def alpha(A,b,x):
    d = df(A,b,x)
    return np.dot(d.T,d)/(np.dot(np.dot(d.T,A),d))

xnew = np.array([[0],[0]])

for time in range(0,10):
    xold = xnew
    a = alpha(A1,b1,xold)
    d = a * df(A1,b1,xold)
    xnew = xold -d
    if np.abs(max(d)) < 0.001:
        break

print(time)
print(xnew)
