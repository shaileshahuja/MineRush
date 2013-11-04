__author__ = 'Shailesh'

from sklearn.ensemble import RandomForestClassifier
import utils
from sklearn import cross_validation
from collections import defaultdict
import operator
import numpy as np
import math
import logging
import datetime

def addFeatures(part, width):
    features = []
    for multiple in xrange(1, len(part) / width):
        prevPt = (multiple - 1) * width
        currPt = multiple * width
        features.append(np.mean(part[prevPt: currPt]))
        features.append(np.std(part[prevPt: currPt]))
        features.append((part[currPt - 1] - part[prevPt]) / width)
    return features

def preprocess(data):
    splitPt = 2048
    width = 32
    Pdata = [[] for _ in xrange(len(data))]
    for i, sample in enumerate(data):
        lx = sample[:splitPt]
        rx = sample[splitPt:splitPt*2]
        ly = sample[splitPt*2:splitPt*3]
        ry = sample[splitPt*3:]
        parts = [lx, rx, ly, ry]
        for part in parts:
            Pdata[i].extend(addFeatures(part, width))
    return Pdata

def main():
    logging.info("Features: Raw data")
    X, Y = utils.read_data("../files/train.csv")
    #X = preprocess(X)
    X = [x[1:] for x in X]
    Y = [int(x) for x in Y]
    X, Y = np.array(X), np.array(Y)
    classMap = sorted(list(set(Y)))
    accs = []
    rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, oob_score=True)
    logging.info(rf)
    folds = 5
    stf = cross_validation.StratifiedKFold(Y, folds)
    logging.info("CV Folds: " + str(folds))
    loss = []
    for i, (train, test) in enumerate(stf):
        X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
        rf.fit(X_train, y_train)
        predicted = rf.predict(X_test)
        probs = rf.predict_proba(X_test)
        probs = [[min(max(x, 0.001), 0.999) for x in y]
                       for y in probs]
        loss.append(utils.logloss(probs, y_test, classMap))
        accs.append(utils.accuracy(predicted, y_test))
        logging.info("Accuracy(Fold {0}): ".format(i) + str(accs[len(accs) - 1]))
        logging.info("Loss(Fold {0}): ".format(i) + str(loss[len(loss) - 1]))
    logging.info("Mean Accuracy: " + str(np.mean(accs)))
    logging.info("Mean Loss: " + str(np.mean(loss)))

def setupLogging():
    logging.basicConfig(filename='evaluation.log', format='%(message)s', level=logging.DEBUG)
    console = logging.StreamHandler()  # Add the log message handler to the logger
    console.setLevel(logging.INFO)  # set a format which is simpler for console use
    formatter = logging.Formatter('%(message)s')  # tell the handler to use this format
    console.setFormatter(formatter)  # add the handler to the root logger
    logging.getLogger('').addHandler(console)

if __name__ == "__main__":
    setupLogging()
    logging.info("Start at: " + str(datetime.datetime.now()))
    main()
    logging.info("End at: " + str(datetime.datetime.now()))
    logging.info("\n")