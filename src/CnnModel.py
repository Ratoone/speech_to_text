import os
from pathlib import Path

import keras
import numpy as np
import sklearn.model_selection
import tqdm
from keras import Input, Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization  # create model

from src.PreprocessingUtils import PreprocessingUtils


def generate_model(input_shape, output_length):
    input_layer = Input(shape=input_shape)
    layer = BatchNormalization()(input_layer)
    for i in range(4):
        layer = Conv2D(16 * (2 ** i), kernel_size=3, activation="relu")(layer)
        layer = BatchNormalization()(layer)
        layer = MaxPooling2D(2, 2)(layer)

    layer = Flatten()(layer)
    layer = Dense(256, activation="relu")(layer)
    layer = Dropout(0.3)(layer)
    layer = Dense(output_length, activation="softmax")(layer)

    return Model(inputs=input_layer, outputs=layer)


def get_model(input_shape=None, output_length=None):
    model_path = "speech_{}classes_model.json".format(output_length)
    weights_path = "speech_{}classes_weights.h5".format(output_length)
    if os.path.exists(model_path):
        with open(model_path, "r") as f:
            model = keras.models.model_from_json(f.read())
    else:
        model = generate_model(input_shape, output_length)
        with open(model_path, "w") as f:
            f.write(model.to_json())
    if os.path.exists(weights_path):
        model.load_weights(weights_path)

    # compile model using accuracy to measure model performance
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def train_model(x_data: np.ndarray, y_data: np.ndarray, output_size):
    y_data = keras.utils.to_categorical(y_data)
    x_data = x_data.reshape(*x_data.shape, 1)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_data, y_data, test_size=0.2)

    model = get_model(x_data[0].shape, output_size)
    # train the model
    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              epochs=20,
              batch_size=100,
              callbacks=[EarlyStopping(monitor="val_loss", patience=5, mode="min", min_delta=0.01),
                         ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.1, mode="min", min_delta=0.01),
                         ModelCheckpoint(monitor="val_loss",
                                         filepath="speech_{}classes_weights.h5".format(output_size),
                                         save_best_only=True,
                                         save_weights_only=True)
                         ])


def test_model(x_data, y_data, labels):
    model = get_model(output_length=len(labels))
    x_data = x_data.reshape(*x_data.shape, 1)
    y_pred = model.predict(x_data)
    import json
    y_prob = [float(y_pred[row, index]) for row, index in enumerate(y_data)]
    y_pred = [labels[x] for x in y_pred.argmax(axis=1)]
    y_data = [labels[x] for x in y_data]
    data = {}
    # data["confusion_matrix"] = confusion_matrix
    data["actual_y"] = y_data
    data["expected_y"] = y_pred
    data["likelihood_y"] = y_prob
    with open("data.json", "w") as f:
        json.dump(data, f)


def get_data(lower_limit, upper_limit, used_words):
    selection_sample_rate = 16000
    fourier_window_size = 600
    preprocessing = PreprocessingUtils(discard_short_entries=True)
    all_words = ["bed", "bird", "cat", "dog", "down", "eight", "five", "four", "go", "happy", "house", "left", "marvin",
                 "nine", "no", "off", "on", "one", "right", "seven", "sheila", "six", "stop", "three", "tree", "two",
                 "up", "wow", "yes", "zero"]
    missing_words_count = len(all_words) - len(used_words)
    data_folder = str(Path(os.getcwd()).parent.absolute()) + "/dataset/"
    x_data = []
    y_data = []

    for index, word in enumerate(tqdm.tqdm(all_words, desc="Extracting features")):
        folder_path = data_folder + word
        if word in used_words:
            class_number = used_words.index(word)
            lower_bound = lower_limit
            upper_bound = upper_limit
        else:
            class_number = len(used_words)
            lower_bound = lower_limit / missing_words_count
            upper_bound = upper_limit / missing_words_count

        num_words = 0
        for file in os.listdir(folder_path):
            file_path = folder_path + "/" + file
            if ".npy" in file_path:
                continue

            num_words += 1
            if num_words < lower_bound:
                continue
            if num_words > upper_bound:
                break

            time_series = preprocessing.preprocess(file_path)
            if time_series is None:
                num_words -= 1
                continue
            spectogram = preprocessing.log_spectogram(time_series,
                                                      selection_sample_rate,
                                                      fourier_window_size // 2,
                                                      fourier_window_size // 4)
            x_data.append(spectogram)
            y_data.append(class_number)

    return np.array(x_data), np.array(y_data)


if __name__ == '__main__':
    words = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    # x_data, y_data = get_data(0, 2000, words)
    # train_model(x_data, y_data, len(words)+1)
    x_data, y_data = get_data(2000, 2500, words)
    test_model(x_data, y_data, words + ['other'])
