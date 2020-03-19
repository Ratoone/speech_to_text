from sklearn.naive_bayes import GaussianNB
from src.FeatureSelector import FeatureSelector
import numpy as np
import random

VALIDATION_RATE = 0.05

data = FeatureSelector().select()

# Decide which data is used as validation data and which data as train data
random.shuffle(data)
split_index = round(VALIDATION_RATE * len(data))
train_data = data[split_index:]
validation_data = data[:split_index]

# Do the training of the Naive Bayes classifier
gnb = GaussianNB()
train_x = np.array([td.input for td in train_data])
train_y = np.array([td.output for td in train_data])
gnb.fit(train_x, train_y)

# Do the validation part
confusion_matrix = dict()
validation_x = np.array([vd.input for vd in validation_data])
actual_y = np.array([vd.output for vd in validation_data])
expected_y = gnb.predict(validation_x)
for actual, expected in zip(actual_y, expected_y):
    confusion_matrix[(actual, expected)] = confusion_matrix.get((actual, expected), 0) + 1

print(confusion_matrix)