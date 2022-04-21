# This script is intended for students to test their 
# trained and exported models against the studentTest200 
# dataset, which was excluded from the model training data.
# It is ill-advised to train using that data as well, as it 
# will limit the student's ability to evaluate their models
# in an unbiased way. There is no overlap whatsoever between 
# the  training set, this studentTest200 set, and the set against 
# which models will be graded.

import keras
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import load_model
import time

# This accepts exported .h5 models
print("Input the full name of the model you wish to test: ")
modelFile = input()
startTime = time.time()
k=10 # number of runs to average
i=0
accuracyList = []
while(i<k):
    #generate dataset
    path_root = "studentTest200"
    batches = ImageDataGenerator().flow_from_directory(directory=path_root, target_size=(64,64), batch_size=200)
    print(batches.class_indices)

    imgs, labels=next(batches)
    imgs = imgs/255
    x_test = imgs
    y_test = labels

    # CNN model
    num_classes=8
    Malware_model = load_model(modelFile)
    Malware_model.summary()


    scores = Malware_model.evaluate(x_test, y_test)
    print('Final CNN accuracy for round ', i+1, ":", scores[1])

    accuracyList.append(scores[1])
    i += 1

#end while increment and all
print("\nEnd of k average loop\n\nAccuracyList:")
endTime = time.time()
min = (endTime - startTime) / 60 # time in minutes

# calc average, max
average=0
max=0
x=0
while(x<k):
    print("Round", x+1, ":", accuracyList[x] * 100)
    average += accuracyList[x]
    if(max < accuracyList[x]):
        max = accuracyList[x]
    x+=1
average = (average / k) * 100
max = max * 100

print("\nFinal average for", k, "runs is", average)
print("Max of", k, "runs is", max)
print("Execution time:", min, "minutes\n")
