__author__ = 'Shailesh'

from sklearn.ensemble import RandomForestClassifier
import utils
from sklearn import cross_validation
from collections import defaultdict
import operator
import numpy as np
import math

def main():
    X, Y = utils.read_data("../files/train.csv")
    X = [x[1:] for x in X]
    Y = [int(x) for x in Y]
    X, Y = np.array(X), np.array(Y)
    classMap = sorted(list(set(Y)))
    accs = []
    rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, oob_score=True)
    stf = cross_validation.StratifiedKFold(Y, 5)
    loss = []
    for train, test in stf:
        X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
        rf.fit(X_train, y_train)
        #predicted = rf.predict(X_test)
        probs = rf.predict_proba(X_test)
        probs = [[min(max(x, 0.001), 0.999) for x in y]
                       for y in probs]
        loss.append(utils.logloss(probs, y_test, classMap))
        #accs.append(utils.accuracy(predicted, y_test))
        #print accs[len(accs) - 1]
        print loss[len(accs) - 1]
    #scores = cross_validation.cross_val_score(rf, X, Y, cv=10)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    #print "Mean Accuracy:", np.mean(accs)
    print "Mean Loss:", np.mean(loss)
if __name__ == "__main__":
    main()