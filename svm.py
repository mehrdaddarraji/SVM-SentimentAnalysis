import pandas as pd
import string
from pandas import DataFrame
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import roc_auc_score
 
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
        if score != 3:
            review = line[index + 1:-7]
            #reviews.append(review)

            if score >= 4:
                reviews.append(review)
                sentiments.append(1)
            else:
                for i in range(1):
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
 
# TFIDF Vectorizer - used to convert reviews from text to features
stopset = set(stopwords.words('english'))
vectorizer = CountVectorizer(analyzer=text_process)
 
# dataframes for train and test dataset
train_df = sentimentDataFrame("lab_train.txt")
X_train, y_train = vectorizer.fit_transform(train_df.reviews), train_df.sentiments
 
test_df = sentimentDataFrame("lab_test.txt")
X_test, y_test = vectorizer.transform(test_df.reviews), test_df.sentiments

# train using linear support vector classification
clf = svm.SVC(probability=True, kernel='linear')
clf.fit(X_train, y_train)
#clf.score(X_train, y_train)
 
# test models accuracy
print ("Model Accuracy: ", roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))
 
# import bookings.com comments
comments_df = pd.read_excel("evaluation_dataset.xlsx", header=None, names=['reviews'])
comments_vector = vectorizer.transform(comments_df['reviews'])
comments_df['sentiments'] = clf.predict(comments_vector)
 
print (comments_df)