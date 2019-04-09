#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
start_time_voting_ensemble= time.time()

import pickle
import numpy as np
import pandas as pd
import codecs

from sklearn import tree
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn import model_selection
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#The paths
TRAIN_DATASET_PATH='train_dataset'
TRAIN_LABELS_PATH='train_labels'

TEST_DATASET_PATH='test_dataset'
TEST_LABELS_PATH='test_labels'

PRED_LABELS_PATH='pred'    
MODEL_PATH='model'

#importing train dataset
print("loading train and test datas")

with codecs.open(TRAIN_DATASET_PATH,'r',encoding='utf8',errors='ignore') as tweets_train_file:
    train_tweets = tweets_train_file.read().splitlines()
with open(TRAIN_LABELS_PATH) as labels_train_file:
    train_labels = labels_train_file.read().splitlines()

#importing test dataset
with codecs.open(TEST_DATASET_PATH,'r',encoding='utf8',errors='ignore') as tweets_test_file:
    test_tweets = tweets_test_file.read().splitlines()
with open(TEST_LABELS_PATH) as labels_test_file:
    test_labels = labels_test_file.read().splitlines()

print("End of loading datsets")

seed=7
num_trees = 1
kfold = model_selection.KFold(n_splits=20, random_state=seed)
estimators = []

#starting classifing with 6 classifiers

print("doing SGDClassifier model")
clf1 = Pipeline([('vect1', CountVectorizer()),
                ('tfidf1', TfidfTransformer()),
                ('clf1', SGDClassifier(
                loss='modified_huber',
                penalty='l2',
                alpha=1e-4, 
                random_state=0,
                max_iter=300,
                n_jobs=-1,
                average=False,
                class_weight=None,
                epsilon=1.0,
                eta0=0.0,
                fit_intercept=True,
                l1_ratio=0.15,
                learning_rate='optimal',
                shuffle=True,
                tol=None,
                verbose=0,
                warm_start=False
                )),])
estimators.append(('SGDClassifier', clf1))
clf1 =clf1.fit(train_tweets,train_labels)

print("doing RandomForestClassifier model")
clf2 = Pipeline([('vect2', CountVectorizer()),
                     ('tfidf2', TfidfTransformer()),
                     ('clf2', RandomForestClassifier(n_estimators=num_trees, random_state=seed)),])
estimators.append(('RandomForestClassifier', clf2))
clf2 =clf2 .fit(train_tweets,train_labels)

print("doing DecisionTreeClassifier model")
clf3 = Pipeline([('vect3', CountVectorizer()),
                 ('tfidf3', TfidfTransformer()),
                 ('clf3', DecisionTreeClassifier()),])
estimators.append(('DecisionTreeClassifier', clf3))
clf3 =clf3 .fit(train_tweets,train_labels)

print("doing BernoulliNB model")
clf4 = Pipeline([('vect4', CountVectorizer()),
                 ('tfidf4', TfidfTransformer()),
                 ('clf4', BernoulliNB(alpha=1,fit_prior=True,class_prior=None)),])
estimators.append(('BernoulliNB', clf4))
clf4 =clf4 .fit(train_tweets,train_labels)

print("doing SVM model")
clf5 = Pipeline([('vect5', CountVectorizer()),
                 ('tfidf5', TfidfTransformer()),
                 ('clf5', svm.SVC(kernel='linear', C=0.5 ,gamma= 0.10,probability=True)),])
estimators.append(('SVM', clf5))
clf5 =clf5 .fit(train_tweets,train_labels)

print("doing Extra Tree model")
clf6 = Pipeline([('vect6', CountVectorizer()),
                 ('tfidf6', TfidfTransformer()),
                 ('clf6', ExtraTreesClassifier(n_estimators=num_trees, max_features=300)),])
estimators.append(('Extra Tree', clf6))
clf6 =clf6 .fit(train_tweets,train_labels)

print ("voting")
ensemble = VotingClassifier(estimators,voting='soft')
ensemble.fit(train_tweets,train_labels).predict(test_tweets)

print("Evaluation of the results")
diff = np.setdiff1d(train_labels, test_labels)
if diff:
    raise ValueError("test_labels contains new labels: %s" % str(diff))
    results = model_selection.cross_val_score(ensemble, train_tweets, train_labels, cv=kfold)
    print(results.mean())

print("doing estimations")

labels_pred= ensemble.predict(test_tweets)

print ("confusion matrix", confusion_matrix(test_labels,labels_pred,labels=['pos','neg']))
print ("Accuracy: ", accuracy_score(test_labels,labels_pred)*100)
print ("précision: " , (precision_score(test_labels, labels_pred,average = None) ))

print("Saving the model")
open_write_model = open(MODEL_PATH, 'wb')
pickle.dump((ensemble), open_write_model)
open_write_model.close()

print("writing the estimations")
write_predictions=open(PRED_LABELS_PATH,"w")
for polarity in labels_pred:
    write_predictions.write(polarity+"\n")
    
interval_voting_ensemble = time.time() - start_time_voting_ensemble
print ('End of the program')
print ('Total time in seconds for voting:', interval_voting_ensemble)
