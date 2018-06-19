import numpy as np
import matplotlib.pyplot as plt
import random
import csv

learn_rate = 0.0001
learn_time = 1000
soft_margin = 100

data = np.loadtxt("myteach.csv", delimiter= ",")
test = np.loadtxt("mytest.csv", delimiter= ",")
#print(data)
row , col = data.shape

def svm(x, y):
    R = np.dot(x.T ,x)
    A = np.zeros((row, row))
#    print("R",R)


    for i in range(row):
        for j in range(row):
            A[i][j] = y[i]*y[j]*R[i][j]

#    print("A",A)

    O = np.ones((row,1))


    a = np.zeros((row,1))
    dL = -1 * np.dot(A,a) + O
    k=0
    while(k<learn_time):
        for l in range(row):
            
            a[l,0] = a[l,0] + learn_rate * dL[l,0]
            
            if a[l,0] < 0:
                a[l,0] = 0
                continue
            
            if a[l,0] > soft_margin:
                a[l,0] = soft_margin
                continue
        dL = -1 * np.dot(A,a) + O
        k += 1

#    print("a0=")
#    print(a)
    psi = np.dot(x,a*y)
#    print("psi", psi)

    psi = np.array(psi)

    return psi

def judgement(test, psi):
    f = open("iijima_.csv", "w")
    writer = csv.writer(f, lineterminator='\n')
    for id, i  in enumerate(test, start = 1):
        judge = np.dot(i,psi)
        
        if judge < 0:
            print("spam")
            writer.writerow([id, "spam"])

        elif judge > 0:
            print("ham")
            writer.writerow([id, "ham"])
        else:
            if random.random() >= 0.5:
                print("spam")
                writer.writerow([id, "spam"])
            else:
                print("ham")
                writer.writerow([id, "ham"])
    

    f.close()

x = data[:, :col-1]
x = np.hstack((x, np.ones((row, 1))))

x = x.T

y = np.matrix(data[:, col-1]).T
y[y==0] = -1
#print(x)
#print(y)
y = np.array(y)

psi = svm(x ,y)


test = np.hstack((test, np.ones((len(test),1))))
#print(test)

judgement(test, psi)
