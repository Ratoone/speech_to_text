from sklearn.decomposition import PCA
from src.FeatureExtractor import FeatureExtractor
from typing import NamedTuple, List
import gc
import math
import numpy as np
import os
import sklearn.feature_selection
import random
import sklearn.feature_selection

Datarow = NamedTuple("Datarow", [("input", np.array), ("output", str)])
class FeatureSelector:
    # All words in the dataset which will be used as training data
    SELECT_WORDS = ["bed", "bird", "cat", "dog", "down", "eight", "five", "four", "go", "happy", "house", "left",
                    "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three",
                    "tree", "two", "up", "wow", "yes", "zero"]

    # All words that we try to recognize with our machine learning algorithm
    RECOGNIZE_WORDS = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"}

    # Critical p-value for Anova F Feature selection (indicates how much information this feature gives about the label).
    # Every feature that has a higher p-value will be discarded.
    SELECTION_P_VALUE = 1e-3

    # Take a number of principal components until this relative amount of variance is explained by this PCA.
    PCA_EXPLAINED_VARIANCE = 0.95

    # The sample used for feature selection
    FEATURE_SELECTION_SAMPLE = 0.1

    @classmethod
    def select(self) -> List[Datarow]:
        # Get a list of all files that contain data
        recognize_files = []
        non_recognize_files = []
        for word in self.SELECT_WORDS:
            recognized_word = word if word in self.RECOGNIZE_WORDS else ""
            folder_path = FeatureExtractor.DATA_FOLDER + word
            for file in os.listdir(folder_path):
                file_path = folder_path + "/" + file
                if file_path.endswith(".npy"):
                    if recognized_word == "":
                        non_recognize_files.append(file_path)
                    else:
                        recognize_files.append((file_path, recognized_word))

        # Do class balancing
        non_recognize_files = random.sample(non_recognize_files, len(recognize_files) // len(self.RECOGNIZE_WORDS))

        # Load all the data
        data = []
        for rf in recognize_files:
            data.append(Datarow(np.load(rf[0]), rf[1]))
        for nrf in non_recognize_files:
            data.append(Datarow(np.load(nrf), ""))

        # Logarithmic transformation
        for i, d in enumerate(data):
            log_trans = [math.log(1 + max(x, 0)) for x in d.input]
            data[i] = Datarow(np.array(log_trans), d.output)
        print("Total features: " + str(len(data[0].input)))

        # Anova F Feature selection preparation
        X = np.array([d.input for d in data])
        y = np.array([d.output for d in data])
        _, p_values = sklearn.feature_selection.f_classif(X, y)
        selection_indices = [i for i, p in enumerate(p_values) if p <= self.SELECTION_P_VALUE]
        print("Features after ANOVA F-selection: " + str(len(selection_indices)))
        for i, d in enumerate(data):
            data[i] = Datarow(np.take(d.input, selection_indices), d.output)

        # Principal Component Analysis preparation
        X = np.array([d.input for d in data])
        pca = PCA(n_components=self.PCA_EXPLAINED_VARIANCE, svd_solver="full")
        pca.fit(X)
        print("Features after PCA: " + str(len(X[0])))
        gc.collect()
        for i, d in enumerate(data):
            data[i] = Datarow(pca.transform(np.array([d.input]))[0], d.output)

        return data