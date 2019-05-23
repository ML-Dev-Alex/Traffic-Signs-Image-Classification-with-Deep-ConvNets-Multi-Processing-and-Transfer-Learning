import helpers
import random
import cv2
import logging
import os
import pandas as pd
import numpy as np
from tensorflow.python.keras.models import model_from_json
from tensorflow.python import ConfigProto
from tensorflow.python import InteractiveSession


# Suppress tensorflow warnings
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Allow tensorflow/cuda to allocate memory as it needs, rather than upfront (which causes some problems)
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Define variables
folder = '/home/alexandre/Documents/Data/TrafficSigns/'

test_dataframe = pd.read_csv('/home/alexandre/Documents/Data/TrafficSigns/Test.csv')

class_list = list(test_dataframe['ClassId'].value_counts().sort_index().index)

number_of_images = len(test_dataframe.index)


def predict(model_name, architecture_name, image_size, is_gray):
    # Set random seed for reproducibility
    random.seed(123)

    X_test = np.zeros((number_of_images, image_size, image_size, 3), dtype=np.float32)
    y_test = []

    augmenter = helpers.get_augmenter()

    with open(f'{architecture_name}.json', 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(f'{model_name}.hdf5')

    for i in range(number_of_images):
        current_image_path = f"{folder}/{test_dataframe['Path'].iloc[i]}"
        current_image = cv2.imread(current_image_path)
        current_image = cv2.resize(current_image, (image_size, image_size))/255
        if is_gray:
            current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
            for j in range(3):
                if j == 0:
                    X_test[i, :, :, 0] = current_image
                elif j == 1:
                    X_test[i, :, :, 1] = augmenter.augment_image(current_image)
                elif j == 2:
                    X_test[i, :, :, 2] = augmenter.augment_image(current_image)
        else:
            X_test[i, :, :, :] = current_image

        y_test.append(test_dataframe['ClassId'].iloc[i])

    predictions = model.predict(X_test, batch_size=4)
    predictions = predictions.argmax(axis=-1)
    return predictions, y_test


if __name__ == '__main__':
    model_name = 'Xception_color_weights'
    architecture_name = 'Xception_color_architecture'
    image_size = 299

    predictions, y_test = predict(model_name=model_name, architecture_name=architecture_name, image_size=image_size,
                                  is_gray=False)

    print(f"\nColor Xception top 1 accuracy = {(np.count_nonzero(predictions == y_test)/number_of_images):.2f},\n",
          file=open('predictions.txt', 'a'))









