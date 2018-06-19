import numpy as np
import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn import mixture
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("spam.csv",header=0, encoding = "latin-1" )
data.replace("ham",0)
data = data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)

data["v1"] = data.v1.map({"ham":0, "spam":1})

X_train ,X_test ,Y_train ,Y_test = train_test_split(data["v2"] ,data["v1"] ,test_size=0.2 ,random_state = 5)

vectorizer = TfidfVectorizer(use_idf=True)
vectorizer.fit(X_train)
X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print X_train_tfidf.toarray().shape

pca = PCA(n_components=50)
pca.fit(X_train_tfidf.toarray())
X_train_pca = pca.transform(X_train_tfidf.toarray())
pca.fit(X_test_tfidf.toarray())
X_test_pca = pca.transform(X_test_tfidf.toarray())

print X_train_pca.shape

gmm = mixture.GaussianMixture(n_components=2,covariance_type='full',random_state=20)
gmm.fit(X_train_pca)

result = gmm.predict(X_test_pca)
print(accuracy_score(Y_test,result))

spamcnt=0
hamcnt=0
rhamcnt=0
rspamcnt=0

good=0
bad=0

spam_ham=0
ham_spam=0

for i in Y_test:
    if i==0:
        hamcnt += 1
    else:
        spamcnt += 1

for i in result:
    if i==0:
        rhamcnt += 1
    else:
        rspamcnt += 1
j=0

for i in Y_test:
    if result[j]==i:
        good += 1
    else:
        bad += 1
        if i==0:
            ham_spam+=1
        else:
            spam_ham+=1
    j += 1

print(hamcnt)
print(spamcnt)
print(rhamcnt)
print(rspamcnt)

print(ham_spam)
print(spam_ham)

print(good)
print(bad)
print(float(good)/(good+bad))

'''
colors = ['r' if i==0 else 'g' for i in Y_train]

axe = Axes3D(plt.figure())
axe.scatter(X_train_pca[:,0],X_train_pca[:,1],X_train_pca[:,2],c=colors)
plt.show()
'''
