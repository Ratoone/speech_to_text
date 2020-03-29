import math
import os
import multiprocessing.process
from pathlib import Path
from typing import List

import numpy as np
import scipy as sc
import scipy.stats
import tqdm as tqdm

from src.PreprocessingUtils import PreprocessingUtils


class FeatureExtractor:
    # The folder with all sound fragments
    DATA_FOLDER = str(Path(os.getcwd()).parent.absolute()) + "/dataset/"

    # All words in the dataset for which we have sound fragments
    ALL_WORDS = ["bed", "bird", "cat", "dog", "down", "eight", "five", "four", "go", "happy", "house", "left", "marvin",
                 "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two",
                 "up", "wow", "yes", "zero"]

    # Only files with this sample rate are used as training and validation data (which basically are all files except 1)
    SELECTION_SAMPLE_RATE = 16000

    # All the quantiles used as features
    QUANTILES = [0.0, 0.03125, 0.0625, 0.125, 0.25, 0.5, 0.75, 0.875, 0.9375, 0.96875, 1.0]

    # All the trimmed means as features
    TRIMMED_MEANS = [0.015625, 0.0625, 0.25]

    # The size of every fourier transform window
    FOURIER_WINDOW_SIZE = 600

    # How much the fourier window is shifted everytime
    FOURIER_WINDOW_SHIFTS = 10

    # Apply a max on the amplitudes in every group of these sizes
    FREQUENCY_MERGE_STEPS = 5

    forced_quit = False

    preprocessing = PreprocessingUtils(expected_sample_rate=SELECTION_SAMPLE_RATE)

    def run(self, number_of_parallel_jobs=1):
        pool = multiprocessing.Pool(processes=number_of_parallel_jobs)
        for _ in tqdm.tqdm(pool.imap_unordered(self.extract_files, self.ALL_WORDS), total=len(self.ALL_WORDS), desc="Extracting features for words"):
            pass

    def extract_files(self, word):
        folder_path = self.DATA_FOLDER + word
        for file in os.listdir(folder_path):
            file_path = folder_path + "/" + file
            self.extract_file_if_possible(file_path)

    def extract_file_if_possible(self, file_path: str):
        if file_path.endswith(".wav"): # Check if the file is not a npy file
            npy_file_path = file_path.replace(".wav", ".npy")
            if not os.path.isfile(npy_file_path): # Only extract the file if it hasn't yet been extracted
                time_series = self.preprocessing.preprocess(file_path)
                if time_series is not None:
                    self.extract(time_series, npy_file_path.replace(".npy", ""))

    def extract(self, time_series: List[float], npy_file_path: str):
        spectogram = self.preprocessing.log_spectogram(time_series,
                                                       self.SELECTION_SAMPLE_RATE,
                                                       self.FOURIER_WINDOW_SIZE * 2,
                                                       self.FOURIER_WINDOW_SIZE * 2 - self.FOURIER_WINDOW_SHIFTS)
        features = self.preprocessing.max_pooling(spectogram, (1, self.FREQUENCY_MERGE_STEPS))
        stats = self.statistics(np.array(features))
        np.save(npy_file_path, stats)

    def statistics(self, matrix: np.array) -> np.array:
        features = []
        features.extend(np.mean(matrix, axis=0))
        features.extend(np.std(matrix, axis=0))
        features.extend(sc.stats.skew(matrix, axis=0))
        features.extend(sc.stats.kurtosis(matrix, axis=0))
        features.extend(np.quantile(matrix, self.QUANTILES, axis=0).flatten())
        for tm in self.TRIMMED_MEANS:
            features.extend(sc.stats.trim_mean(matrix, tm, axis=0).flatten())

        return np.array(features)


if __name__ == "__main__":
    FeatureExtractor().run(8)