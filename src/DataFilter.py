from scipy.stats import stats
from src.DataParser import Raw_data
from typing import List
import numpy as np

class DataFilter:
    # The minimum z-score that occurs for the main sound fragment. Noise should mostly be below this z-score.
    MAIN_SOUND_Z_SCORE = 1.0

    # The z-score at which the difference between pitches (moments that are above the z-score) will be cut off
    # (considered to be different sound parts).
    PITCH_DIFFERENCE_Z_SCORE = 3.0

    # Minimum occurrences of pitches. If the number of occurrences is lower then it will be left unfiltered.
    MINIMUM_PITCH_OCCURRENCES = 400

    @classmethod
    def filter(self, all_data: List[Raw_data]):
        filtered_data = []
        for data in all_data:
            z_scores = stats.zscore(data.time_series)
            pitch_indices = [i for i, z_score in enumerate(z_scores) if z_score >= self.MAIN_SOUND_Z_SCORE]
            if len(pitch_indices) < self.MINIMUM_PITCH_OCCURRENCES:
                filtered_data.append(data)
                continue

            pitch_differences = [j - i for i, j in zip(pitch_indices, pitch_indices[1:])]
            mean_pitch_diff = np.mean(pitch_differences)
            std_pitch_diff = np.std(pitch_differences)
            cut_off_pitch_diff = mean_pitch_diff + self.PITCH_DIFFERENCE_Z_SCORE * std_pitch_diff
            largest_part = self.get_largest_part(0, cut_off_pitch_diff, pitch_indices, [])
            start_index = largest_part[0]
            end_index = largest_part[-1] + 1
            filtered_data.append(Raw_data(data.word, data.time_series[start_index:end_index]))

    @classmethod
    def get_largest_part(self, start_index: int, cut_off_pitch_diff: float, pitch_indices: List[int],
                         best_part: List[int]) -> List[int]:
        i = 0
        part = [pitch_indices[start_index]]
        for i in range(start_index, len(pitch_indices) - 1):
            difference = pitch_indices[i + 1] - pitch_indices[i]
            if difference > cut_off_pitch_diff:
                break
            part.append(pitch_indices[i + 1])

        best_part = part[:] if len(part) > len(best_part) else best_part[:]
        if i >= len(pitch_indices) - 2:
            return best_part
        else:
            return self.get_largest_part(i + 1, cut_off_pitch_diff, pitch_indices, best_part)