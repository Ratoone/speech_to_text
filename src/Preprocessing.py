from typing import Optional

import librosa
import numpy as np


class Preprocessing:
    """
    Class for all your preprocessing needs. Will perform resampling and (maybe) filtering
    """

    def __init__(self, expected_sample_rate: int = 16000, desired_sample_rate: int = None, discard_short_entries: bool = False):
        """
        Initialize the Preprocessing object
        :param expected_sample_rate: the expected sample rate of the audio files; the ones that do not match are discarded
        :param desired_sample_rate: the resampling sample rate; if not specified, no resampling will be performed
        :param discard_short_entries: the normal entry length is 1 second; specify whether to discard the shorter ones or not
        """
        self.expected_sample_rate = expected_sample_rate
        self.desired_sample_rate = desired_sample_rate
        self.discard_short_entries = discard_short_entries
        self.duration = 1  # the time series all have one second

    def preprocess(self, file_path: str) -> Optional[np.ndarray]:
        """
        Reads the input wav file and returns a processed signal
        :param file_path: the full path of the file
        :return: the time series resampled to the desired sample rate, or none in some conditions (see class description)
        """
        time_series, sample_rate = librosa.load(file_path, sr=None)
        if sample_rate != self.expected_sample_rate:
            return None

        if self.discard_short_entries and len(time_series) < self.expected_sample_rate * self.duration:
            return None

        if self.desired_sample_rate:
            time_series = librosa.resample(time_series, sample_rate, self.desired_sample_rate)
            sample_rate = self.desired_sample_rate

        return time_series


