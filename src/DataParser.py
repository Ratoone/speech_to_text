from pathlib import Path
from typing import NamedTuple, List, Union
import math
import numpy as np
import os
import scipy.io.wavfile

Raw_data = NamedTuple("Raw_Data", [("word", Union[str, None]), ("time_series", np.array)])
class DataParser:
    # The folder with all sound fragments.
    DATA_FOLDER = str(Path(os.getcwd()).parent.absolute()) + "/dataset/"

    # All words in the dataset for which we have sound fragments.
    ALL_WORDS = ["bed", "bird", "cat", "dog", "down", "eight", "five", "four", "go", "happy", "house", "left", "marvin",
                 "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two",
                 "up", "wow", "yes", "zero"]

    # All words that we try to recognize with our machine learning algorithm.
    RECOGNIZE_WORDS = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"}

    # Only files with this sample rate are used as training and validation data (which basically are all files except 1)
    SELECTION_SAMPLE_RATE = 16000

    @classmethod
    def parse(self) -> List[Raw_data]:
        data = []
        min_size = math.inf
        for word in self.ALL_WORDS:
            folder_path = self.DATA_FOLDER + word
            recognize_word = word if word in self.RECOGNIZE_WORDS else None
            for file in os.listdir(folder_path):
                file_path = folder_path + "/" + file
                sample_rate, time_series = scipy.io.wavfile.read(file_path)
                if sample_rate == self.SELECTION_SAMPLE_RATE:
                    min_size = min(min_size, len(time_series))
                    data.append(Raw_data(recognize_word, time_series))
        print(min_size)
        return data