import pandas as pd
from pandas import DataFrame
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
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
           
            if score >= 4:
                reviews.append(review)
                sentiments.append(1)
            else:
                # append negative reviews more since they are scarce
                # in the data set
                for i in range(7):
                    reviews.append(review)
                    sentiments.append(0)
           
    # creating DataFrame out of text and sentiment
    df = DataFrame({'reviews':reviews, 'sentiments':sentiments})
    return df
 

# TFIDF Vectorizer - used to convert reviews from text to features
stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True,
                            strip_accents='ascii', stop_words=stopset)
 
# dataframes for train and test dataset
train_df = sentimentDataFrame("lab_train.txt")
X_train, y_train = vectorizer.fit_transform(train_df.reviews), train_df.sentiments
 
test_df = sentimentDataFrame("lab_test.txt")
X_test, y_test = vectorizer.transform(test_df.reviews), test_df.sentiments

# train using linear support vector classification
clf = svm.SVC(probability=True, kernel='linear', degree=1, gamma=1)
clf.fit(X_train, y_train)
 
# test models accuracy
print ("Model Accuracy: ", roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))
 
# import bookings.com comments
comments_df = pd.read_excel("evaluation_dataset.xlsx", header=None, names=['reviews'])
comments_vector = vectorizer.transform(comments_df['reviews'])
comments_df['sentiments'] = clf.predict(comments_vector)
 
print (comments_df)