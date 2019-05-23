import cv2
import random
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
from imgaug import augmenters as iaa


def read_bgr_to_rgb(path):
    """Reads images from bgr format with opencv into rgb format."""
    path = str(path)
    bgr_img = cv2.imread(path)
    b, g, r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r, g, b])
    return rgb_img


def visualize_images(path, class_list, dataframe, id_feature_name, path_feature_name, save_file, num_columns=3,
                     class_names=None):
    """
    Visualize images in a matplotlib.pyplot grid.
    :param path: Path of the folder with images to be displayed.
    :param class_list: List of image classes to be displayed.
    :param dataframe: Pandas dataframe containing 'paths' and 'ids' of images to be displayed.
    :param id_feature_name: Name of column containing image 'ids' in the dataframe.
    :param path_feature_name: Name of column containing 'image paths' in the dataframe.
    :param save_file: Filename to save pyplot figure in.
    :param num_columns: Number of images per class to be displayed, defaults to 3.
    :param class_names: List with names of classes, if None is given, class_names = class_list.
    """
    if class_names is None:
        class_names = class_list

    num_classes = len(class_names)
    fig, axes = plt.subplots(nrows=num_classes, ncols=num_columns, dpi=200, figsize=(num_columns, num_classes))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    for i in range(num_classes):
        current_class = dataframe.loc[dataframe[id_feature_name] == class_list[i]]
        for j in range(num_columns):
            # Select random images from the class to show.
            image_name = current_class[path_feature_name].iloc[random.randint(0, len(current_class) - 1)]
            image_path = f'{path}/{image_name}'
            axes[i, j].imshow(read_bgr_to_rgb(path=image_path))
            axes[i, j].axis('off')
            axes[i, j].set_title(class_names[i], color='red', fontsize=8)

    if save_file is not None:
        plt.savefig(save_file)


def visualize_percentages(class_list, dataframe, class_feature_name, save_file):
    """
    Visualize the proportion of classes in a dataframe.
    :param class_list: List with names of classes.
    :param dataframe: Pandas dataframe.
    :param class_feature_name: Name of column containing the classes.
    :param save_file: Filename to save pyplot figure in.
    """
    _, ax = plt.subplots(nrows=1, ncols=1, dpi=100, figsize=(len(class_list), 6))

    image = sns.countplot(dataframe[class_feature_name], order=dataframe[class_feature_name].value_counts().index,
                          palette='Accent')
    image.set_title('class proportions in percentages', color='green')
    image.set_xticklabels(labels=class_list, fontsize=8)

    # Print percentage above bar.
    for patch in ax.patches:
        height = patch.get_height()
        width = patch.get_width()
        # Center on bar and put it 1% above the top of the bar.
        ax.text(x=patch.get_x() + width / 2, y=height + height / 100,
                s=f'{100 * height / len(dataframe):.2f}%', ha='center')

    if save_file is not None:
        plt.savefig(save_file)


def preprocess_class_grey(images_per_class, image_size, current_class, current_class_dataframe,
                          current_class_index, path_feature_name, folder, q):
    """
    Creates a vector and populates it with preprocessed images from an image class.
    :param images_per_class: Number of images to store in vector per class.
    :param image_size: Height and Width of a square image to be pre-processed.
    :param current_class: ID of the Current class.
    :param current_class_dataframe: Dataframe containing the current class.
    :param current_class_index: Index of the image to be processed.
    :param path_feature_name: Name of the feature containing the path to the images in the dataframe.
    :param folder: Path of the folder in which images are stored.
    :param q: Queue to store variables and pass them through the multiple processes.
    :return: Returns a vector of shape (images_per_class, image_size, image_size) with type float32 containing
    the pre-processed images.
    """
    class_vector = np.zeros((images_per_class, image_size, image_size, 3), dtype=np.float32)
    augmenter = get_augmenter()

    images_in_class = len(current_class_dataframe)
    for i in range(images_per_class):
        image_number = current_class_index % images_in_class
        current_class_index += 1

        current_image_path = f'{folder}/{current_class_dataframe[path_feature_name].iloc[image_number]}'
        current_image = cv2.imread(current_image_path)
        current_image = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        current_image = cv2.resize(current_image, (image_size, image_size))
        for j in range(3):
            if j == 0:
                class_vector[i, :, :, 0] = current_image
            elif j == 1:
                class_vector[i, :, :, 1] = augmenter.augment_image(current_image)
            elif j == 2:
                class_vector[i, :, :, 2] = augmenter.augment_image(current_image)

    q.put((np.copy(class_vector / 255), current_class, current_class_index))


def populate_vector_grey(x_array, y_list, images_per_class, number_of_classes, image_size,
                         class_dataframes, class_indexes, folder_path, path_feature_name):
    """
    Use every CPU core to load pre-processed images into an array (does not work on jupyter notebook).
    :param x_array: Array in which to store images.
    :param y_list: List of correct labels in same order as they are stored in the array.
    :param images_per_class: Number of images to store per class.
    :param number_of_classes: Number of different classes in the dataframe.
    :param image_size: Size of Width and Height of images in the dataframe.
    :param class_dataframes: List with dataframes containing only images from each class.
    :param class_indexes: List that stores index numbers for each class, in order to use every image in each class
    sequentially in multiple processes.
    :param folder_path: Path of the folder in which images are stored.
    :param path_feature_name: Column in which image paths are stored in the dataframes.
    :return: x_array with preprocessed images, y_list with their labels, and class_indexes which is another list with
    the updated indexes per class.
    """
    current_class = 0
    processes = []

    q = mp.Queue()
    while current_class < number_of_classes:
        current_class_dataframe = class_dataframes[current_class]
        process = (mp.Process(target=preprocess_class_grey, args=(images_per_class, image_size, current_class,
                                                                  current_class_dataframe, class_indexes[current_class],
                                                                  path_feature_name, folder_path, q,)))
        processes.append(process)
        for i in range(images_per_class):
            y_list.append(current_class)
        current_class += 1

    for process in processes:
        process.start()

    for process in processes:
        process.join(timeout=1)

    for i in range(number_of_classes):
        temp = q.get()
        current_class = temp[1]
        x_array[current_class * images_per_class:current_class * images_per_class + images_per_class, :, :, :] = temp[0]
        class_indexes[current_class] = temp[2]

    return x_array, y_list, class_indexes


def preprocess_class_color(images_per_class, image_size, current_class, current_class_dataframe,
                           current_class_index, path_feature_name, folder, q, variations):
    """
    Creates a vector and populates it with preprocessed images from an image class.
    :param images_per_class: Number of images to store in vector per class.
    :param image_size: Height and Width of a square image to be pre-processed.
    :param current_class: ID of the Current class.
    :param current_class_dataframe: Dataframe containing the current class.
    :param current_class_index: Index of the image to be processed.
    :param path_feature_name: Name of the feature containing the path to the images in the dataframe.
    :param folder: Path of the folder in which images are stored.
    :param q: Queue to store variables and pass them through the multiple processes.
    :param variations: Number of stochastic variations per image.
    :return: Returns a vector of shape (images_per_class, image_size, image_size) with type float32 containing
    the pre-processed images.
    """
    class_vector = np.zeros((images_per_class * variations, image_size, image_size, 3), dtype=np.float32)
    augmenter = get_augmenter()

    images_in_class = len(current_class_dataframe)
    for i in range(images_per_class):
        image_number = current_class_index % images_in_class
        current_class_index += 1

        current_image_path = f'{folder}/{current_class_dataframe[path_feature_name].iloc[image_number]}'
        current_image = cv2.imread(current_image_path)
        current_image = cv2.resize(current_image, (image_size, image_size))
        for j in range(variations):
            if j == 0:
                class_vector[i*variations, :, :, :] = current_image
            else:
                class_vector[i*variations + j, :, :, :] = augmenter.augment_image(current_image)

    q.put((np.copy(class_vector / 255), current_class, current_class_index))


def populate_vector_color(x_array, y_list, images_per_class, number_of_classes, image_size, class_dataframes,
                          class_indexes, folder_path, path_feature_name, variations):
    """
    Use every CPU core to load pre-processed images into an array (does not work on jupyter notebook).
    :param x_array: Array in which to store images.
    :param y_list: List of correct labels in same order as they are stored in the array.
    :param images_per_class: Number of images to store per class.
    :param number_of_classes: Number of different classes in the dataframe.
    :param image_size: Size of Width and Height of images in the dataframe.
    :param class_dataframes: List with dataframes containing only images from each class.
    :param class_indexes: List that stores index numbers for each class, in order to use every image in each class
    sequentially in multiple processes.
    :param folder_path: Path of the folder in which images are stored.
    :param path_feature_name: Column in which image paths are stored in the dataframes.
    :param variations: Number of stochastic variations per image.
    :return: x_array with preprocessed images, y_list with their labels, and class_indexes which is another list with
    the updated indexes per class.
    """
    current_class = 0
    processes = []

    q = mp.Queue()
    while current_class < number_of_classes:
        current_class_dataframe = class_dataframes[current_class]
        process = (mp.Process(target=preprocess_class_color, args=(images_per_class, image_size, current_class,
                                                                   current_class_dataframe,
                                                                   class_indexes[current_class],
                                                                   path_feature_name, folder_path, q, variations,)))
        processes.append(process)
        for i in range(images_per_class*variations):
            y_list.append(current_class)
        current_class += 1

    for process in processes:
        process.start()

    for process in processes:
        process.join(timeout=1)

    for i in range(number_of_classes):
        temp = q.get()
        current_class = temp[1]
        x_array[current_class * images_per_class * variations:
                current_class * images_per_class * variations + images_per_class * variations, :, :, :] = temp[0]
        class_indexes[current_class] = temp[2]

    return x_array, y_list, class_indexes


def get_augmenter():
    """
    Defines a sequential image augmenter to pre-process images randomly.
    :return: Returns sequential image augmenter to pre-process images randomly.
    """

    def sometimes(aug):
        return iaa.Sometimes(0.5, aug)

    augmenter = iaa.Sequential(
        [
            # Apply the following augmenters to most images.
            iaa.Fliplr(0.01),
            iaa.Flipud(0.01),
            sometimes(iaa.Affine(
                scale={"x": (0.6, 1.4), "y": (0.6, 1.4)},
                translate_percent={"x": (-0.4, 0.4), "y": (-0.4, 0.4)},
                rotate=(-45, 45),
                shear=(-25, 25),
                order=[0, 1],
                # mode=ia.ALL
            )),

            # Execute 0 to 8 of the following augmenters per image,
            # don't execute all of them, as that would often be way too strong.
            iaa.SomeOf((0, 8),
                       [
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 0.5)),
                               iaa.AverageBlur(k=(1, 3)),
                               iaa.MedianBlur(k=(1, 3)),
                           ]),
                           iaa.Sharpen(alpha=(0.0, 0.5), lightness=(0.5, 1.2)),
                           iaa.Emboss(alpha=(0.0, 0.5), strength=(0, 0.8)),
                           iaa.SimplexNoiseAlpha(iaa.OneOf([
                               iaa.EdgeDetect(alpha=(0.0, 0.3)),
                               iaa.DirectedEdgeDetect(alpha=(0.0, 0.2), direction=(0.0, 1.0)),
                           ])),
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01 * 255)),
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.05)),
                               iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02)),
                           ]),
                           iaa.Add((-40, 40)),
                           iaa.OneOf([
                               iaa.Multiply((0.9, 1.3)),
                               iaa.FrequencyNoiseAlpha(
                                   exponent=(-1, 0),
                                   first=iaa.Multiply((0.9, 1.4)),
                                   second=iaa.ContrastNormalization((0.9, 1.4))
                               )
                           ]),
                           sometimes(iaa.ElasticTransformation(alpha=(0, 0.5), sigma=0.25)),
                           sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                           sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1))),
                       ],
                       random_order=True
                       )
        ],
        random_order=True
    )
    return augmenter
