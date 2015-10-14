import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import re, nltk        
from nltk.stem.porter import PorterStemmer
import random
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack,hstack
from numpy import array

train_data_df = pd.read_csv('trainset.csv', header=None, delimiter="\t", quoting=3)
test_data_df = pd.read_csv('testset.csv', header=None,delimiter="\t" , quoting=3 )

train_data_df.columns = ["Popularity","Body"]
test_data_df.columns = ["Body"]

print train_data_df.shape
print test_data_df.shape

print train_data_df.Popularity.value_counts()

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    
    text = re.sub("[^a-zA-Z]", " ", text)
    text = re.sub("[http://]", " ", text)
    tokens = nltk.word_tokenize(text)
    
    stems = stem_tokens(tokens, stemmer)
    return stems

vectorizer = TfidfVectorizer(analyzer = 'word',tokenizer = tokenize,lowercase = True,stop_words = 'english',max_features = 85)
vectorizer1 = TfidfVectorizer(analyzer = 'word',tokenizer = tokenize,lowercase = True,stop_words = 'english',max_features = 5)
vectorizer2 = TfidfVectorizer(analyzer = 'word',tokenizer = tokenize,lowercase = True,stop_words = 'english',max_features = 3)


corpus_data_features = vectorizer.fit_transform(train_data_df.Body.tolist() + test_data_df.Body.tolist())
corpus_data_features1 = vectorizer1.fit_transform(train_data_df.Title.tolist() + test_data_df.Title.tolist())
corpus_data_features2 = vectorizer2.fit_transform(train_data_df.Tags.tolist() + test_data_df.Tags.tolist())

corpus_data_features_nd = corpus_data_features.toarray()
corpus_data_features_nd1 = corpus_data_features1.toarray()
corpus_data_features_nd2 = corpus_data_features2.toarray()

corpus_data_features_nd = corpus_data_features_nd * 0.2
corpus_data_features_nd1 = corpus_data_features_nd1 * 0.5
corpus_data_features_nd2 = corpus_data_features_nd2 * 0.3

body_title_corpus = []
body_title_tags_corpus = []

for i in range(len(corpus_data_features_nd)):

    body_title_corpus.append(np.concatenate((corpus_data_features_nd[i], corpus_data_features_nd1[i])))

body_title_corpus = array(body_title_corpus)

for i in range(len(corpus_data_features_nd)):
    body_title_tags_corpus.append(np.concatenate((body_title_corpus[i], corpus_data_features_nd2[i])))

body_title_tags_corpus = array(body_title_tags_corpus)

print body_title_tags_corpus    

X_train, X_test, y_train, y_test  = train_test_split(body_title_tags_corpus[0:len(train_data_df)], train_data_df.Popularity, random_state=2) 

print body_title_tags_corpus.shape

logreg_model = LogisticRegression(penalty = 'l1')
   
logreg_model = logreg_model.fit(X=X_train, y=y_train)

#y_pred = logreg_model.predict(X_test)
y_pred = logreg_model.predict(X_test)

# get predictions

accu_score = cross_val_score(logreg_model,X_test,y_pred,cv=10,scoring='accuracy').mean()
print "\n"
print "accuracy score : ",accu_score  

precision_score = cross_val_score(logreg_model,X_test,y_pred,cv=10,scoring='precision').mean()
#print "\n"
print "precision score : ",precision_score

recall_score = cross_val_score(logreg_model,X_test,y_pred,cv=10,scoring='recall').mean()
#print "\n"
print "recall score : ",recall_score
  
f1_score = cross_val_score(logreg_model,X_test,y_pred,cv=10,scoring='f1').mean()
#print "\n"
print "f1 score : ",f1_score,"\n"


# train classifier

logreg_model = LogisticRegression(penalty = 'l1')

logreg_model = logreg_model.fit(X=body_title_tags_corpus[0:len(train_data_df)], y=train_data_df.Popularity)
test_pred = logreg_model.predict(body_title_tags_corpus[len(train_data_df):])

# sample some of them

spl = random.sample(xrange(len(test_pred)),10)
    

for text, Popularity in zip(test_data_df.Title[spl], test_pred[spl]):
    print Popularity,"====", text,"\n"
