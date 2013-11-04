__author__ = 'Shailesh'


import utils
import numpy as np
from sklearn import cross_validation

def main():
    X, Y = utils.read_data("../files/train_10.csv")
    n_target = len(set(Y))
    Y = map(int, Y)
    folds = 5
    stf = cross_validation.StratifiedKFold(Y, folds)
    loss = []
    accs = []
    classMap = sorted(list(set(Y)))
    X, Y = np.array(X), np.array(Y)
    print "Testing..."
    for i, (train, test) in enumerate(stf):
        X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
        probs = [[0.001 for x in range(n_target)]
                           for y in range(len(y_test))]
        loss.append(utils.logloss(probs, y_test, classMap))
        accs.append(utils.accuracy([1]*len(y_test), y_test))
        print "Accuracy(Fold {0}): ".format(i) + str(accs[len(accs) - 1])
        print "Loss(Fold {0}): ".format(i) + str(loss[len(loss) - 1])
    print "Mean Accuracy: " + str(np.mean(accs))
    print "Mean Loss: " + str(np.mean(loss))
if __name__== "__main__":
    main()