mport numpy as np
import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt


XA = np.array([[-5,0],[1,4],[-2,1],[-2,3]],dtype = float)
XB = np.array([[2,-1],[2,-3],[5,0],[-1,-4]],dtype = float)
uA = np.zeros((2,1))
uB = np.zeros((2,1))

uA[0,0] = np.mean(XA[:,0])
uA[1,0] = np.mean(XA[:,1])

uB[0,0] = np.mean(XB[:,0])
uB[1,0] = np.mean(XB[:,1])



XA[:,0] -= uA[0,0]
XA[:,1] -= uA[1,0]
  
XB[:,0] -= uB[0,0]
XB[:,1] -= uB[1,0]

CA = (1/4) * np.dot(XA.T,XA)
CB = (1/4) * np.dot(XB.T,XB)

sigmaW = 0.5*(CA + CB)
uAB = np.array([[uA[0,0],uA[1,0]],[uB[0,0],uB[1,0]]])

sigmaB = 0.5*(np.dot(uAB,uAB.T))
sigmaWinv = np.linalg.inv(sigmaW)
sigmaWinv_B = np.dot(np.linalg.inv(sigmaW),sigmaB)

print("uA =\n" , uA)
print("uB =\n" , uB)
print("CA =\n" , CA)
print("CB =\n" , CB)
print("sigmaW =\n", sigmaW)
print("sigmaB =\n", sigmaB)

print("sigmaWinv =\n", sigmaWinv)
print("sigmaWinv_B\n",sigmaWinv_B)

la, v = np.linalg.eig(sigmaWinv_B)
print("lambda,v =\n", la,"\n", v)
XA = np.array([[-5,0],[1,4],[-2,1],[-2,3]],dtype = float)
XB = np.array([[2,-1],[2,-3],[5,0],[-1,-4]],dtype = float)

x_fig = np.array(range(-5,5))
y_fig = v[1,1]/v[1,0]*x_fig
plt.scatter(XA[:,0],XA[:,1],c="blue")
plt.scatter(XB[:,0],XB[:,1],c="red")
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x_fig,y_fig)
plt.show()

