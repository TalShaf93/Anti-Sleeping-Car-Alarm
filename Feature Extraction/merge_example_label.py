#################################################
#               Imported Libraries              #
#################################################
import csv
import numpy as np

#################################################
#                 Read features                 #
#################################################
with open('features.csv') as file:  # check if video has been seen before
    features = csv.reader(file)
    features = list(features)

#################################################
#                  Read labels                  #
#################################################
with open('training.csv') as f:  # check if video has been seen before
        labels = csv.reader(f)
        labels = list(labels)

#################################################
#     Write features and labels to new file     #
#################################################
combined = []
for feature in features:
    checked = False
    with open('merge.csv') as file:  # check if video has been seen before
        reader = csv.reader(file)
        for row in reader:
            if feature[0] in row[0]:
                checked = True
                break     
    for label in labels:
        if label[0] == feature[0] and label[1] == feature[1] and not checked:
           combined.append([label[0],label[1],feature[2],feature[3],feature[4],feature[5],feature[6],feature[7],feature[8],feature[9],feature[10],feature[11],label[2]])
if not combined:
    print('everything merged')
else:
    print('merged successfully')

##########################################################
#                   add all data to csv                  #
##########################################################
with open('merge.csv','a',newline='') as f:  # add all data to csv
    thewriter = csv.writer(f)
    for example in combined:
        thewriter.writerow(example)