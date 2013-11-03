__author__ = 'Shailesh'

from sklearn import svm
import utils

def main():
    training, target = utils.read_data("../files/train.csv")
    training = [x[1:] for x in training]
    target = map(float, target)
    test, _ = utils.read_data("../files/test.csv")
    test = [x[1:] for x in test]

    svc = svm.SVC(probability=True)
    svc.fit(training, target)
    predicted_probs = svc.predict_proba(test)
    predicted_probs = [[min(max(x,0.001),0.999) for x in y]
                       for y in predicted_probs]
    predicted_probs = [["%f" % x for x in y] for y in predicted_probs]
    utils.write_delimited_file("../files/svm_benchmark.csv",
                                predicted_probs)

if __name__ == "__main__":
    main()