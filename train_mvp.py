import os
import logging
import random
import helpers
import numpy as np
import pandas as pd
from tensorflow.python.keras.applications import Xception
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python import ConfigProto
from tensorflow.python import InteractiveSession

# Suppress tensorflow warnings
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Allow tensorflow/cuda to allocate memory as it needs, rather than upfront (which causes some problems)
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

if __name__ == '__main__':
    # Set random seed for reproducibility
    random.seed(123)

    # Define variables/hyper parameters
    folder = '/home/alexandre/Documents/Data/TrafficSigns/'

    train_dataframe = pd.read_csv('/home/alexandre/Documents/Data/TrafficSigns/Train.csv')

    model_name = 'Xception_grey_weights'

    class_list = list(train_dataframe['ClassId'].value_counts().sort_index().index)

    number_of_classes = len(class_list)

    image_size = 72

    number_of_cores = 4

    batch_size = 32
    epochs = 4
    batches = 10

    train_images_per_class = 1000
    val_images_per_class = 200

    # Separate the full dataframe into class dataframes and append them to a list.
    class_dataframes = []
    for image_class_number in class_list:
        image_class = train_dataframe.loc[train_dataframe['ClassId'] == image_class_number]
        class_dataframes.append(image_class)

    # Create a list with the current image index number of each class, to pass to our multi-processing function.
    # (I wanted to use generators initially, but since they can't be pickled, and therefore are not passable to other
    # processes, I decided to do this instead).
    class_indexes = []
    for i in range(number_of_classes):
        class_indexes.append(0)

    # Create a checkpoint to save the best weights as the model trains.
    checkpoint = ModelCheckpoint(filepath=f'{model_name}.hdf5', monitor='val_acc', verbose=1, save_best_only=True,
                                 save_weights_only=True, mode='max')

    # We will use the Xception model architecture, but not its pre-trained weights, as it would require us to have
    # color images of size 299x299x3 to be able to transfer learning from its learned parameters.
    model = Xception(weights=None, include_top='false', pooling='max', input_shape=(image_size, image_size, 3))

    x = model.output
    x = Dense(128, activation="relu")(x)
    predictions = Dense(number_of_classes, activation='softmax')(x)

    model = Model(model.input, predictions)

    # Save model architecture only once, to be able to save only weights during training.
    with open('Xception_grey_architecture.json', 'w') as f:
        f.write(model.to_json())

    model.compile(optimizer=Adam(lr=1e-3), loss='sparse_categorical_crossentropy', metrics=['acc'])

    if os.path.isfile(f'{model_name}.hdf5'):
        model.load_weights(f'{model_name}.hdf5')
        print('Model loaded.')

    for i in range(batches):
        print(f'\nStarting batch {i+1}.\n')
        # Create the vectors that will nest all images, and the lists with their labels.
        X_train = np.zeros((number_of_classes * train_images_per_class, image_size, image_size, 3), dtype=np.float32)
        y_train = []

        X_val = np.zeros((number_of_classes * val_images_per_class, image_size, image_size, 3), dtype=np.float32)
        y_val = []

        # Populate class vectors with [images_per_class, image_size, image_size, 3] images, where the other 'channel'
        # dimensions contain stochastic pre-processed variations of the original greyscale image (stored on [:, :, 0]).
        # We then concatenate these vectors into a single vector with the data from all classes.
        X_train, y_train, class_indexes = helpers.populate_vector_grey(x_array=X_train, y_list=y_train, folder_path=folder,
                                                                       class_dataframes=class_dataframes,
                                                                       number_of_classes=number_of_classes,
                                                                       images_per_class=train_images_per_class,
                                                                       image_size=image_size,
                                                                       class_indexes=class_indexes,
                                                                       path_feature_name='Path')

        X_val, y_val, class_indexes = helpers.populate_vector_grey(x_array=X_val, y_list=y_val, folder_path=folder,
                                                                   class_dataframes=class_dataframes,
                                                                   number_of_classes=number_of_classes,
                                                                   images_per_class=val_images_per_class,
                                                                   image_size=image_size,
                                                                   class_indexes=class_indexes,
                                                                   path_feature_name='Path')

        model.fit(X_train, np.asarray(y_train), validation_data=(X_val, np.asarray(y_val)),
                  batch_size=batch_size, epochs=epochs, verbose=1, callbacks=[checkpoint])
