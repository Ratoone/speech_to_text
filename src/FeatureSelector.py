from src.FeatureExtractor import FeatureExtractor
from typing import NamedTuple, Union, List
import numpy as np
import os
import sklearn.feature_selection

Datarow = NamedTuple("Datarow", [("input", np.array), ("output", Union[str, None])])
class FeatureSelector:
    # All words in the dataset which will be used as training data
    SELECT_WORDS = ["no", "yes"]

    SELECTION_P_VALUE = 1e-2

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

        X = np.array([d.input for d in data])
        y = np.array([d.output for d in data])
        _, p_values = sklearn.feature_selection.f_classif(X, y)
        selection_indices = [i for i, p in enumerate(p_values) if p <= self.SELECTION_P_VALUE]
        for i, d in enumerate(data):
            data[i] = Datarow(np.take(d.input, selection_indices), d.output)

        return data