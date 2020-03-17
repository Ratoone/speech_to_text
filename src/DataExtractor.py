from numpy import fft
from src.DataParser import Raw_data
from typing import List
import numpy as np

class DataExtractor:
    @classmethod
    def extract(self, filtered_data: List[Raw_data]):
        yes_data = []
        no_data = []
        for data in filtered_data:
            if data.word == "yes":
                yes_data.append(fft.fft(data.time_series, 400))
            if data.word == "no":
                no_data.append(fft.fft(data.time_series, 400))
        yes_data = np.array(yes_data)
        no_data = np.array(no_data)
        yes_mean = np.mean(yes_data, axis=0)
        no_mean = np.mean(no_data, axis=0)
        yes_std = np.std(yes_data, axis=0)
        no_std = np.std(no_data, axis=0)
        for ym, ystd, nm, nstd in zip(yes_mean, yes_std, no_mean, no_std):
            print("Yes-mean: " + str(ym))
            print("Yes-std: " + str(ystd))
            print("No-mean: " + str(nm))
            print("No-std: " + str(nstd))
