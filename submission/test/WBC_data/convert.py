training_csv = open("wdbc-train-csv.data", "wb") 
testing_csv = open("wdbc-test-csv.data", "wb") 

with open("wdbc.names", "rb") as headerfile:
    headerfile.next()
    headers = [ x.rstrip() for x in headerfile.next().split(",")]

training_csv.write("class, ")
for i in xrange(len(headers)):
    training_csv.write(headers[i])
    training_csv.write(",")
training_csv.write("\n")

testing_csv.write("class, ")
for i in xrange(len(headers)):
    testing_csv.write(headers[i])
    testing_csv.write(",")
testing_csv.write("\n")

with open("wdbc-train.data", "rb") as infile:
    for line in infile:
        lines = [ x.rstrip() for x in line.split("\r")]
        for l in lines:
            features_and_class = l.split(",")
            training_csv.write(features_and_class[-1])
            training_csv.write(",")
            for i in xrange(len(features_and_class)-1):
                training_csv.write(features_and_class[i])
                if i < len(features_and_class) - 2:
                    training_csv.write(",")
            training_csv.write("\n")

with open("wdbc-test.data", "rb") as infile:
    for line in infile:
        lines = [ x.rstrip() for x in line.split("\r")]
        for l in lines:
            features_and_class = l.split(",")
            testing_csv.write(features_and_class[-1])
            testing_csv.write(",")
            for i in xrange(len(features_and_class)-1):
                testing_csv.write(features_and_class[i])
                if i < len(features_and_class) - 2:
                    testing_csv.write(",")
            testing_csv.write("\n")

training_csv.close()
testing_csv.close()


