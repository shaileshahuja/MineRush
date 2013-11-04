__author__ = 'Shailesh'

import math

def read_data(file_name):
    f = open(file_name)
    #ignore header
    f.readline()
    samples = []
    target = []
    for line in f:
        line = line.strip().split(",")
        sample = [float(x) for x in line[1:]]
        samples.append(sample)
        target.append(line[0])
    return (samples, target)

def write_delimited_file(file_path, data,header=None, delimiter=","):
    f_out = open(file_path,"w")
    if header is not None:
        f_out.write(delimiter.join(header) + "\n")
    for line in data:
        if isinstance(line, str):
            f_out.write(line + "\n")
        else:
            f_out.write(delimiter.join(line) + "\n")
    f_out.close()

def number_samples_with_class(predicted, label):
    count = 0
    for next in predicted:
        if next == label:
            count+=1
    return count

def number_accurate_with_class(predicted, real, label):
    i = 0
    accurate = 0
    while i < len(predicted):
        if predicted[i] == label:
            if predicted[i] == real[i]:
                accurate+=1
        i+=1
    return accurate

def precision_with_class(predicted, real, label):
    predicted_count = number_samples_with_class(predicted, label)
    accurate = number_accurate_with_class(predicted, real, label)
    if predicted_count == 0:
        predicted_count = 0.001
    return float(accurate) / predicted_count

def recall_with_class(predicted, real, label):
    real_count = number_samples_with_class(real, label)
    accurate = number_accurate_with_class(predicted, real, label)
    if real_count == 0:
        return 0
    return float(accurate) / real_count

def f1_with_class(predicted, real, label):
    precision = precision_with_class(predicted, real, label)
    recall = recall_with_class(predicted, real, label)
    if precision + recall == 0:
        precision = 0.001
        recall = 0.001
    return 2* (precision * recall) / (precision + recall)

def accuracy(predicted, real):
    i = 0
    accurate = 0
    while i < len(predicted):
        if predicted[i] == real[i]:
            accurate+=1
        i += 1
    return float(accurate)/len(real)


def logloss(predicted, real, classMap):
    data = zip(real, predicted)
    probs = [math.log(row[classMap.index(actual)]) for actual, row in data]
    total = sum(probs)
    loss = -1 * total / len(real)
    return loss