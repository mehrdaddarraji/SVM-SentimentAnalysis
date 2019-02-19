import pandas as pd
import string
from pandas import DataFrame
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.metrics import roc_auc_score, confusion_matrix
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE, ADASYN
 
# function to create dataframes for text and sentiment
def sentimentDataFrame(filename):
   
    # import train data
    file = open(filename, "r")
    lines = file.readlines()
    lines.pop(0)
    file.close()
   
    # data structure for text and sentiment
    reviews = []
    sentiments = []
   
    # a DataFrame for the text and the corresponding sentiment
    for line in lines:
       
        index = 0
        char = ''
       
        # find the start of the comment
        while char != ',' and index < len(line):
           
            char = line[index]
            index += 1
       
       
        # score of the the text turned into pos/neg
        score = int(line[-4])
       
        # get the text
        review = line[index + 1:-7]
        #reviews.append(review)

        if score >= 3:
            reviews.append(review)
            sentiments.append(1)
        else:
            reviews.append(review)
            sentiments.append(0)
           
    # creating DataFrame out of text and sentiment
    df = DataFrame({'reviews':reviews, 'sentiments':sentiments})
    return df

def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
 
# Vectorizer - used to convert reviews from text to features
vectorizer = CountVectorizer(analyzer=text_process)
 
# dataframes for train and test dataset
train_df = sentimentDataFrame("lab_train.txt")
X_train, y_train = vectorizer.fit_transform(train_df.reviews), train_df.sentiments
 
test_df = sentimentDataFrame("lab_test.txt")
X_test, y_test = vectorizer.transform(test_df.reviews), test_df.sentiments

sm = ADASYN()
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# train using linear support vector classification
clf = svm.LinearSVC()
clf.fit(X_train_res, y_train_res)
#clf.fit(X_train, y_train)
 
# test models accuracy
print ("Model Accuracy: ", clf.score(X_test, y_test))


predicted_y = clf.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, predicted_y).ravel()
precision_score = tp / (tp + fp)
recall_score = tp / (tp + fn)
print ("Precision Score: ", precision_score)
print ("Recall Score: ", recall_score)
 
# import bookings.com comments
comments_df = pd.read_excel("evaluation_dataset.xlsx", header=None, names=['reviews'])
comments_vector = vectorizer.transform(comments_df['reviews'])
comments_df['sentiments'] = clf.predict(comments_vector)
 
print (comments_df)