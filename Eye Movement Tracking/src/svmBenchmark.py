__author__ = 'Shailesh'

from sklearn import svm, cross_validation
import utils
import numpy as np

def main():
    X, Y = utils.read_data("../files/train_10.csv")
    Y = map(int, Y)
    folds = 5
    stf = cross_validation.StratifiedKFold(Y, folds)
    loss = []
    svc = svm.SVC(probability=True)
    accs = []
    classMap = sorted(list(set(Y)))
    X, Y = np.array(X), np.array(Y)
    print "Testing..."
    for i, (train, test) in enumerate(stf):
        X_train, X_test, y_train, y_test = X[train], X[test], Y[train], Y[test]
        svc.fit(X_train, y_train)
        predicted = svc.predict(X_test)
        probs = svc.predict_proba(X_test)
        probs = [[min(max(x, 0.001), 0.999) for x in y]
                       for y in probs]
        loss.append(utils.logloss(probs, y_test, classMap))
        accs.append(utils.accuracy(predicted, y_test))
        print "Accuracy(Fold {0}): ".format(i) + str(accs[len(accs) - 1])
        print "Loss(Fold {0}): ".format(i) + str(loss[len(loss) - 1])
    print "Mean Accuracy: " + str(np.mean(accs))
    print "Mean Loss: " + str(np.mean(loss))

if __name__ == "__main__":
    main()