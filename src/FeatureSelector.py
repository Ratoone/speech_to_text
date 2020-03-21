from sklearn.decomposition import PCA
from src.FeatureExtractor import FeatureExtractor
from typing import NamedTuple, Union, List
import numpy as np
import os
import sklearn.feature_selection

Datarow = NamedTuple("Datarow", [("input", np.array), ("output", Union[str, None])])
class FeatureSelector:
    # All words in the dataset which will be used as training data
    SELECT_WORDS = ["no", "yes"]

    # Critical p-value for Anova F Feature selection (indicates how much information this feature gives about the label).
    # Every feature that has a higher p-value will be discarded.
    SELECTION_P_VALUE = 1e-2

    # Take a number of principal components until this relative amount of variance is explained by this PCA.
    PCA_EXPLAINED_VARIANCE = 0.99

    # All words that we try to recognize with our machine learning algorithm
    RECOGNIZE_WORDS = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"}

    @classmethod
    def select(self) -> List[Datarow]:
        data = []
        for word in self.SELECT_WORDS:
            folder_path = FeatureExtractor.DATA_FOLDER + word
            for file in os.listdir(folder_path):
                file_path = folder_path + "/" + file
                if file_path.endswith(".npy"):
                    data.append(Datarow(np.load(file_path), word))

        # Anova F Feature selection
        X = np.array([d.input for d in data])
        y = np.array([d.output for d in data])
        _, p_values = sklearn.feature_selection.f_classif(X, y)
        selection_indices = [i for i, p in enumerate(p_values) if p <= self.SELECTION_P_VALUE]
        for i, d in enumerate(data):
            data[i] = Datarow(np.take(d.input, selection_indices), d.output)

        # Principal Component Analysis
        X = np.array([d.input for d in data])
        pca = PCA(n_components=0.99, svd_solver="full")
        pca.fit(X)
        for i, d in enumerate(data):
            data[i] = Datarow(pca.transform(np.array([d.input]))[0], d.output)

        # Polynomial Expansion
        return data