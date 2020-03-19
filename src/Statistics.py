from src.DataExtractor import DataExtractor
import numpy as np
import os

def get_data(word: str):
    results = []
    folder_path = DataExtractor.DATA_FOLDER + word
    for file in os.listdir(folder_path):
        file_path = folder_path + "/" + file
        if file_path.endswith(".npy"):
            results.append(np.load(file_path))
    return np.array(results)

yes_data = get_data("yes")
no_data = get_data("no")
yes_mean = np.mean(yes_data, axis=0)
no_mean = np.mean(no_data, axis=0)
yes_std = np.std(yes_data, axis=0)
no_std = np.std(no_data, axis=0)
for ym, ystd, nm, nstd in zip(yes_mean, yes_std, no_mean, no_std):
    print("Yes mean: " + str(ym))
    print("Yes std: " + str(ystd))
    print("No mean: " + str(nm))
    print("No std: " + str(nstd))
