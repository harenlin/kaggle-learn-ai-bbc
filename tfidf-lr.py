import pandas as pd
import numpy as numpy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def label_to_number(label):
    for i in range(len(label)):
        if label[i] == 'tech': label[i] = 1
        elif label[i] == 'business': label[i] = 2
        elif label[i] == 'sport': label[i] = 3
        elif label[i] == 'entertainment': label[i] = 4
        elif label[i] == 'politics': label[i] = 5
    return label

def number_to_label(ans):
    ans_list = []
    for i in range(0,len(ans)):
        if ans[i] == 1: ans_list.append('tech')
        elif ans[i] == 2: ans_list.append('business')
        elif ans[i] == 3: ans_list.append('sport')
        elif ans[i] == 4: ans_list.append('entertainment')
        elif ans[i] == 5: ans_list.append('politics')
    return ans_list

def makecsv(id_test, test_y_pred):
    #change the ans label
    test_y_pred = number_to_label(test_y_pred)
    submission = {'ArticleId': id_test, 'Category': test_y_pred}
    submission = pd.DataFrame(submission)
    submission.to_csv('submission.csv', index = 0, header = 1)
    
# read data
train_corpus = pd.read_csv('./train.csv')
test_corpus = pd.read_csv('./test.csv')

# training data
x_train = train_corpus['Text']
y_train = train_corpus['Category']
y_train = label_to_number(y_train)

# testing data
x_test = test_corpus['Text']
id_test = test_corpus['ArticleId']

# TFIDF
TFIDF_vectorizer = TfidfVectorizer(stop_words='english')
train_X_TFIDF    = TFIDF_vectorizer.fit_transform( x_train )
test_X_TFIDF     = TFIDF_vectorizer.transform( x_test )
# logistic regression
print('TFIDF + logistic regression')
classifier = LogisticRegression().fit( train_X_TFIDF, y_train.astype('int') )
# ".astype('int')" deal with ValueError: Unknown label type: 'unknown'
#print("the score of the model is "
#       +str(classifier.score( train_X_TFIDF, y_train.astype('int'))))
pred_TFIDF = classifier.predict(test_X_TFIDF)
print( pred_TFIDF )

# make submission file
makecsv(id_test, pred_TFIDF)
