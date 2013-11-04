__author__ = 'Shailesh'

with open("../files/sampleNormalized.txt") as handle, open("../files/sampleNormalized2.txt", 'w') as output:
    for line in handle:
        line = line.strip()[1:-1]
        data = line.split(",")
        for elem in data:
            output.write(elem.strip() + "\n")
        output.write("\n\n\n")

