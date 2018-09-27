import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
import os
import sys
from datetime import datetime


train_data = pd.read_csv('train_set.csv')
test_data = pd.read_csv('test_set.csv')
# test_id = test["id"].copy()
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, use_idf=1, smooth_idf=1, sublinear_tf=1)
trn_term_doc = vec.fit_transform(train_data["word_seg"])
test_term_doc = vec.transform(test_data["word_seg"])

y = (train_data["class"]-1).astype(int)
lin_clf = svm.LinearSVC()
lin_clf.fit(trn_term_doc, y)
predictions = lin_clf.predict(test_term_doc)
with open('baseline.csv', 'w', encoding='utf-8') as f:
    i = 0
    f.write("id,class"+"\n")
    for item in predictions:
        f.write(str(i)+","+str(item+1)+"\n")
        i += 1