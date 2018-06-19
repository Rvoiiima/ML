"""
    Logistic regression and create only spam datasets for LSTM.
    
    References
    Arun Prakash, "Text Preprocessing and Machine Learning Modeling"
    on https://www.kaggle.com/futurist/text-preprocessing-and-machine-learning-modeling
    
    madisonmay, CommonRegex https://github.com/madisonmay/CommonRegex
    
    
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from commonregex import phone
from commonregex import email
from commonregex import link
from commonregex import price
from commonregex import phones_with_exts
from commonregex import date
from commonregex import time
from commonregex import CommonRegex
from sklearn.metrics import confusion_matrix
def preprocessor(text):
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)\(|D|P)',text)
    text = re.sub('[\W]+', ' ', text.lower()) + ''.join(emoticons).replace('-', '')
    return text

    

data = pd.read_csv("spam.csv",header=0, encoding = "latin-1" )
data.replace("ham",0)
data["v3"] = data.v1.map({"ham":0, "spam":1})
data = data.drop(["Unnamed: 2","Unnamed: 3","Unnamed: 4"],axis=1)

write_text = data[data.v3 == 1]["v2"].apply(preprocessor)

data["v2"] = data["v2"].apply(preprocessor)
print(data.head(3))
write_text.to_csv("spam_text.txt",index=False)
fd = open("spam_text.txt","rt" )
spamtext = fd.read()

"""
parser  = CommonRegex()
print(parser.phones(spamtext))
for phone in parser.phones(spamtext):
    spamtext.replace(phone, "")

for link in parser.links(spamtext):
    spamtext.replace(link, "")

for email in parser.emails(spamtext):
    spamtext.replace(email, "")
fd.close()
"""
spamtext = re.sub(email, " ", spamtext)
spamtext = re.sub(link, " ", spamtext)
spamtext = re.sub(phone, " ", spamtext)
spamtext = re.sub(price, " ", spamtext)
spamtext = re.sub(phones_with_exts, " ", spamtext)
#spamtext = spamtext.lower()
spamtext = re.sub('(\w|\d){13,100}', '', spamtext)

"""
re.sub(date, " ", spamtext)
re.sub(time, " ", spamtext)
"""
fd2 = open("spam_fixed3.txt",mode = "wt")
fd2.write(spamtext)
fd2.close()




X_train ,X_test ,Y_train ,Y_test = train_test_split(data["v2"] ,data["v1"] ,test_size=0.2 ,random_state = 5)

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

vector = CountVectorizer()
vector.fit(X_train)

X_train_df = vector.transform(X_train)
Y_train_df = vector.transform(Y_train)
X_test_df = vector.transform(X_test)
print(X_test_df)
LR = LogisticRegression()
LR.fit(X_train_df,Y_train)



Y_predict = LR.predict(X_test_df)
print(accuracy_score(Y_test,Y_predict))

print(confusion_matrix(Y_test,Y_predict))
