import numpy as np

alpha =
tol = 0.1

def heuristic():



def examineExample(i2):
    y2 = target[i2]
    alph2 = alpha[i2]
    E2 = svm(point[i2]) - y2
    r2 = E2*y2

    Not0C = filter(lambda x: 0<x<C, alpha)

    if (r2 < -tol and alph2 < C) or ( r2 > tol and alph2 > 0):

        if len(Not0C):
            i1 = heuristic()
            if takeStep(i1, i2):
                return 1

        for i, a in random.shuffle(enumerate(Not0C))
            i1 = i
            if takeStep(i1, i2):

                 return 1

        for i, a in random.shuffle(enumerate(alpha)):
            i1 = i
            if takeStep(i1, i2):
                return 1

    return 0



if __name__ == "__main__":
    C = 5
    x = np.zeros(8)
    x = x.T
    numChanged = 0
    examineAll = 1

    while (numChanged > 0) or  (examineAll):
        numChanged = 0
        if examineAll:
            for I in range(x):
                numChanged += examineExample(x[I])
        else:
            for I in range(x):
                if ( alpha[I] != 0 ) and ( alpha{I} != C ):
                    numCHanged += examineExample(x[I])

        if (examineAll == 1):
            examineAll = 0
        elif (numChanged == 0):
            examineAll = 1
