from pathlib import Path
from typing import List
import numpy as np
import os
import scipy as sc
import scipy.stats
import scipy.io.wavfile
import threading

class DataExtractor:
    # The folder with all sound fragments
    DATA_FOLDER = str(Path(os.getcwd()).parent.absolute()) + "/dataset/"

    # All words in the dataset for which we have sound fragments
    ALL_WORDS = ["bed", "bird", "cat", "dog", "down", "eight", "five", "four", "go", "happy", "house", "left", "marvin",
                 "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two",
                 "up", "wow", "yes", "zero"]

    # All words that we try to recognize with our machine learning algorithm
    RECOGNIZE_WORDS = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"}

    # Only files with this sample rate are used as training and validation data (which basically are all files except 1)
    SELECTION_SAMPLE_RATE = 16000

    # All the quantiles used as features
    QUANTILES = [0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 0.875, 0.9375, 0.96875, 1.0]

    # All the trimmed means as features
    TRIMMED_MEANS = [0.015625, 0.0625, 0.25]

    # The maximum possible differencing used
    MAXIMUM_DIFFERENCING = 600

    # The differencing step size (the difference between succeeding and preceeding differencing values)
    DIFFERENCING_STEPS = 10

    forced_quit = False

    @classmethod
    def run(self):
        thread = threading.Thread(target=self.extract_files, args=())
        thread.start()
        input("Press a key to stop the program")
        self.forced_quit = True

    @classmethod
    def extract_files(self):
        for word in self.ALL_WORDS:
            folder_path = self.DATA_FOLDER + word
            for file in os.listdir(folder_path):
                file_path = folder_path + "/" + file
                self.extract_file_if_possible(file_path)
                if self.forced_quit:
                    return
        os._exit(os.EX_OK)

    @classmethod
    def extract_file_if_possible(self, file_path: str):
        if file_path.endswith(".wav"): # Check if the file is not a bin file
            bin_file_path = file_path.replace(".wav", ".npy")
            if not os.path.isfile(bin_file_path): # Only extract the file if it hasn't yet been extracted
                sample_rate, time_series = scipy.io.wavfile.read(file_path)
                if sample_rate == self.SELECTION_SAMPLE_RATE:
                    self.extract(time_series, bin_file_path.replace(".npy", ""))

    @classmethod
    def extract(self, time_series: List[float], bin_file_path: str):
        features = self.statistics(time_series)
        for difference in range(self.DIFFERENCING_STEPS, self.MAXIMUM_DIFFERENCING + 1, self.DIFFERENCING_STEPS):
            try:
                statistics = self.statistics(self.differencing(time_series, difference))
                features.extend(statistics)
            except Exception as e:
                return

        np.save(bin_file_path, np.array(features))

    @classmethod
    def differencing(self, time_series: List[float], difference: int) -> List[float]:
        with np.errstate(all = "raise"):
            return [y2 - y1 for y1, y2 in zip(time_series, time_series[difference:])]

    @classmethod
    def statistics(self, time_series: List[float]):
        ts = np.array(time_series)
        features = [np.mean(ts), np.std(ts), sc.stats.skew(ts), sc.stats.kurtosis(ts)]
        features.extend(np.quantile(ts, self.QUANTILES))
        for tm in self.TRIMMED_MEANS:
            features.append(sc.stats.trim_mean(features, tm))
        return features

DataExtractor().run()