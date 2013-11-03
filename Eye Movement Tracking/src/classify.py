__author__ = 'Shailesh'

def classify(filename):
    with open(filename) as handle:
        for line in handle:
            data = line.split(',')
            data = map(float, data)
            maxD = max(data)
            print data.index(maxD), '\t',

if __name__ == "__main__":
    classify("../files/rf_benchmark.csv")
    print
    classify("C:/Users/Shailesh/Desktop/Courses/CSC489/Project/rf_benchmark.csv")