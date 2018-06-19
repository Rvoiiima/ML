#1w142016-0 RyoIijima
import numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
X1 = np.array([[1,1,1,1,1,1,1,1,1,1,1,1],[1.2,1.25,1.3,1.35,1.45,1.55,1.4,1.5,1.6,1.65,1.7,1.75]])
T1 = np.array([[0,0,0,0,0,0,1,1,1,1,1,1]])

def y(X,w):
    return  1.0/(1+(np.exp((-1)*np.dot(w, X))))
#Hesse
def invH(X,w,T):
    H = np.array([[0.0,0.0],[0.0,0.0]])
    for i in range(2):
        for j in range(2):
            H[i,j] = np.sum((-3*y(X,w)*y(X,w)+2*(np.ones(12)+T)*y(X,w)-T)*y(X,w)*(np.ones(12)-y(X,w))*X[i,:]*X[j,:])
    return np.linalg.inv(H)

def dHdw(X,w,T):
    dH = np.zeros(2)
    for i in range(2)
        dH[i] = np.sum((y(X,w)-T)*y(X,w)*(np.ones(12)-y(X,w))*X[i,:])
    return dH

if __name__ == '__main__':    
    wnew = np.array([[0,0]])
    time = 0
    while True :
        wold = wnew
        d = np.dot(invH(X1,wold,T1),dHdw(X1,wold,T1))
        wnew = wold -d
        time += 1
        if(np.linalg.norm(d) < 0.00001):
            break

    print(time)
    print(wnew)
    x_fig = np.array(np.arange(0.5,2.5,0.05))
    y_fig = 1.0/(1+(np.exp((-1)*(wnew[0,1]*x_fig + wnew[0,0]))))

    plt.scatter(X1[1,0:6],T1[0,0:6],c="blue")
    plt.scatter(X1[1,6: ],T1[0,6: ],c="red")
    plt.xlabel("X")
    plt.ylabel("T")
    plt.plot(x_fig,y_fig)
    plt.show()
