#Natural language Processing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
#Import the dataset
dataset= pd.read_csv('wine reviews.csv')
 
#Clean the texts 
import re
import nltk
nltk.download('all')
from nltk.stem.porter import PortStemmer
corpus=[]
for i in range(1,1000):
    review = re.sub('[a-zA-Z]','',dataset['reviews_text'][i])
    review= review.lower()
    review=review.split()
    ps = PortStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review =''.join(review)
    corpus.append(review)
#Using Naive Bayes for Classification
#Create Bag Of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X= cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,24].values

#Create Test Train Split
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25, random_state =0)

#Fitting Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

#predict test resullts
     
y_pred = classifier.predict(X_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
