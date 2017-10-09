import sys
#sys.path.insert(0, '../skfasttext/')
from skfasttext.FastTextClassifier import FastTextClassifier


"""
File paths
"""
train_file="../data/classifier_test.txt"
test_file="../data/classifier_test.txt"

"""
train model
"""

clf=FastTextClassifier()
print("training clf")
clf.fit(train_file)
print("testing clf")
print(clf.predict_proba(test_file,k_best=3))