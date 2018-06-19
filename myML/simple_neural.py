import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
X1 = np.array([[1,1,1,1,1,1,1,1,1,1],[1,2,3,3,4,0,1,2,3,4],[3,2,2,3,1,2,1,1,0,0]])
T1 = np.array([[1,1,1,1,1,0,0,0,0,0]])

def y(X,w):
    return  1.0/(1+(np.exp((-1)*np.dot(w, X))))

def dHdw(X,w,T):
    dH = np.zeros(3)
    dH[0] = np.sum((y(X,w)-T)*y(X,w)*(np.ones(10)-y(X,w))*X[0,:])
    dH[1] = np.sum((y(X,w)-T)*y(X,w)*(np.ones(10)-y(X,w))*X[1,:])
    dH[2] = np.sum((y(X,w)-T)*y(X,w)*(np.ones(10)-y(X,w))*X[2,:])
    
    return dH
def alpha(a):
    return 0.98 * a

a = 0.5
wnew = np.array([[0,0,0]])
for time in range(0,500):
    wold = wnew
    a = alpha(a)
    d = a * dHdw(X1,wold,T1)
    wnew = wold -d
    if(np.linalg.norm(d) < 0.001):
        break

print(time)
print(wnew)
x_fig = np.array(range(0,5))
y_fig = -wnew[0,1]/wnew[0,2]*x_fig -wnew[0,0]/wnew[0,2]
plt.scatter(X1[1,0:5],X1[2,0:5],c="blue")
plt.scatter(X1[1,5:],X1[2,5:],c="red")
plt.xlabel("x2")
plt.ylabel("x3")
plt.plot(x_fig,y_fig)
plt.show()
