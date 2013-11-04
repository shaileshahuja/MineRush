__author__ = 'Shailesh'

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

def preprocess(data):
    processedData = [[] for _ in xrange(len(data))]
    # taken from http://www.kasprowski.pl/emvic/stimFile.txt
    # format (from, to, Xfocus, Yfocus)
    focusPts = [(0, 398, 0, 0), (398, 536, 2048, -2048), (536, 673, 2048, 0), (673, 810, 0, 2048),
                 (810, 947, 0, 0), (947, 1085, -2048, 0), (1085, 1222, 0, -2048),  (1222, 1364, 2048, 2048),
                 (1364, 1496, 0, 0), (1496, 1634, 2048, 0), (1634, 1771, -2048, -2048), (1771, 2048, 0, 0)]
    for i, sample in enumerate(data):
        lx, rx, ly, ry = splitSampleIntoParts(sample)
        for focus in focusPts:
            # change mean to focus point and
            # corner normalization to (-1, 1)
            lxmean = np.mean(lx[focus[0]:focus[1]])
            lx[focus[0]:focus[1]] = [(point - lxmean + focus[2])/2048 for point in lx[focus[0]:focus[1]]]
            rxmean = np.mean(rx[focus[0]:focus[1]])
            rx[focus[0]:focus[1]] = [(point - rxmean + focus[2])/2048 for point in rx[focus[0]:focus[1]]]
            lymean = np.mean(ly[focus[0]:focus[1]])
            ly[focus[0]:focus[1]] = [(point - lymean + focus[3])/2048 for point in ly[focus[0]:focus[1]]]
            rymean = np.mean(ry[focus[0]:focus[1]])
            ry[focus[0]:focus[1]] = [(point - rymean + focus[3])/2048 for point in ry[focus[0]:focus[1]]]
        processedData[i] = lx + rx + ly + ry
    return processedData


def addFeatures(part, width):
    features = []
    u = 0
    for multiple in xrange(1, len(part) / width):
        prevPt = (multiple - 1) * width
        currPt = multiple * width
        features.append(np.mean(part[prevPt: currPt]))
        features.append(np.std(part[prevPt: currPt]))
        v = (part[currPt - 1] - part[prevPt]) / width
        features.append(v)
        a = (v - u) / width
        features.append(a)
        u = v
    return features

def splitSampleIntoParts(sample):
    splitPt = 2048
    lx = sample[:splitPt]
    rx = sample[splitPt:splitPt*2]
    ly = sample[splitPt*2:splitPt*3]
    ry = sample[splitPt*3:]
    return [lx, rx, ly, ry]

def extractFeatures(data):
    width = 32
    Pdata = [[] for _ in xrange(len(data))]
    for i, sample in enumerate(data):
        parts = splitSampleIntoParts(sample)
        for part in parts:
            Pdata[i].extend(addFeatures(part, width))
    return Pdata

def main():
    print "Reading data..."
    X, Y = utils.read_data("../files/train_10.csv")
    print "Preprocessing..."
    X = preprocess(X)
    print "Extracting Features..."
    X = extractFeatures(X)
    Y = [int(x) for x in Y]
    X, Y = np.array(X), np.array(Y)
    classMap = sorted(list(set(Y)))
    accs = []
    rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1, compute_importances=True, oob_score=True)
    rf.fit(X, Y)
    importantFeatures = []
    for x,i in enumerate(rf.feature_importances_):
        print len(rf.feature_importances_)
        print x, i
        if i>np.average(rf.feature_importances_):
            importantFeatures.append(str(x))
    print 'Most important features:', ', '.join(importantFeatures)
    print rf.oob_score_


if __name__ == "__main__":
    main()
