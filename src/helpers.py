# this file contains helpers function that are needed

import glob
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory


def read_data(model="cnn", augment=False, prefetch=False):
    # fetch the data path
    current_dir = Path(__file__).parent.resolve()
    data_dir = current_dir.parent.joinpath("data/")
    image_size = (150, 150)
    color_mode = "grayscale"

    # loading part adapted from: https://keras.io/api/preprocessing/
    # load train data
    train_ds = image_dataset_from_directory(
        directory=f"{data_dir/'seg_train'}",
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        validation_split=0.3,
        subset="training",
        color_mode=color_mode,
        seed=0,
    )

    # use part of training data as validation set to be used for k-fold
    val_ds = image_dataset_from_directory(
        directory=f"{data_dir/'seg_train'}",
        labels="inferred",
        label_mode="categorical",
        validation_split=0.3,
        image_size=image_size,
        subset="validation",
        color_mode=color_mode,
        seed=0,
    )

    # load test data
    test_ds = image_dataset_from_directory(
        directory=f"{data_dir/'seg_test'}",
        labels="inferred",
        label_mode="categorical",
        image_size=image_size,
        color_mode=color_mode,
    )

    # load prediction data, this doesn't have labels
    # so I don't know how to make above code work
    # adapted from the code:
    # https://www.kaggle.com/hammamjoulani/intel-image-classification-cnn#Reading-Images
    x_pred, y_pred = list(), list()
    data_path = str(Path(data_dir).joinpath("seg_pred/").resolve())
    files = glob.glob(pathname=str(data_path + "/*.jpg"))

    for file in files:
        image = cv2.imread(file)
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_array = cv2.resize(grayscale, image_size)
        x_pred.append(list(image_array))

    # convert that list into numpy array
    x_pred = np.asarray(x_pred, dtype=np.float32).reshape((len(x_pred), *image_size, 1))
    pred_ds = tensorflow.data.Dataset.from_tensor_slices(x_pred)

    # augment the data, to make robust machine
    if augment:
        # augment the data, by introducing random flip, rotation etc
        data_augmentation = keras.Sequential(
            [
                layers.Rescaling(1.0 / 255),
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.2),
            ]
        )
        train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

    if prefetch:
        train_ds = train_ds.prefetch(buffer_size=32)
        test_ds = test_ds.prefetch(buffer_size=32)
        pred_ds = pred_ds.prefetch(buffer_size=32)

    # for deep learning, pass image as it is , it learns spatial relationship
    if model in ("cnn", "ann"):
        return train_ds, test_ds, val_ds, None

    # for earlier models, like KNN, SVM, RF, we need to resize
    # extract train, test data, prediction data
    x_train = np.concatenate([x for x, _ in train_ds], axis=0)
    y_train = np.concatenate([y for _, y in train_ds], axis=0)
    x_val = np.concatenate([x for x, _ in val_ds], axis=0)
    y_val = np.concatenate([y for _, y in val_ds], axis=0)
    x_test = np.concatenate([x for x, _ in test_ds], axis=0)
    y_test = np.concatenate([y for _, y in test_ds], axis=0)

    train = {"data": x_train, "labels": y_train}
    test = {"data": x_test, "labels": y_test}
    val = {"data": x_val, "labels": y_val}
    # pred = {"data": x_pred, "labels": None}

    print("\nCompleted reading the dataset.\n")
    return deepcopy(train), deepcopy(test), deepcopy(val), None


# scale the dataset
def get_scaled_dataset(data: Any) -> Any:
    # scaling my data for the better performance (to reduce run time)
    # ref: https://www.geeksforgeeks.org/how-to-scale-pandas-dataframe-columns/
    # using StandardScaler function
    data = StandardScaler().fit_transform(data)

    # returning copy to prevent list mutation related bugs
    return deepcopy(data)


# get reshaped data for different models, adapted from my code for assignment3
def get_reshaped(
    data: np.ndarray, labels: np.ndarray = None, model: str = "cnn"
) -> Union[tuple[Any, Optional[np.ndarray]], tuple[Any, None], tuple[np.ndarray, Any]]:
    # if model is not CNN, means any of KNN, SVM, RF required reshaping
    if not model == "cnn":
        new_shape = data.shape[0], np.prod(data.shape[1:])
        return data.reshape(new_shape).copy(), labels
    else:
        # adopted from keras_mnist.py given to us
        data = data.astype("float32") / 255.0
        if labels is None:
            return data.copy(), np.ndarray([])
    return data, keras.utils.to_categorical(labels, 6)


def plot(model_name, history, validation=True):
    train_accuracy = history["accuracy"]
    plt.figure()

    # if kfold or any other kind of validation is done
    plt.plot(train_accuracy, "b-")

    # if validation is done, we need that plot too
    if validation:
        validation_accuracy = history["val_accuracy"]
        plt.plot(validation_accuracy, "r-")
        plt.title(f"{model_name}, {max(train_accuracy)} {max(validation_accuracy)}")
    else:
        plt.title(f"{model_name}, {max(train_accuracy)}")

    # save to proper directory, # fig
    dir_path = Path(__file__).parent.parent.joinpath("figs/performance/").resolve()
    file_path = dir_path.joinpath(f"{model_name}.png")
    plt.savefig(file_path)

    # plt.show()
    return None


# adapted from: http://www.tarekatwan.com/index.php/2017/12/how-to-plot-a-confusion-matrix-in-python/
def plot_confusion_matrix(cm, classes, title="Confusion Matrix", cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    s = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(s[i][j]) + " = " + str(cm[i][j]))

    # save to proper directory, # fig
    dir_path = Path(__file__).parent.parent.joinpath("figs/performance/").resolve()
    file_path = dir_path.joinpath("cmatrix.png")
    plt.savefig(file_path)
    plt.show()


def get_AUC(ds):
    # for this convert the images in the whole dataset to 3x3 size. So we have
    # 9 pixels, each can be seen as features and do AUC on that.

    cols = ["feature", "auc"]

    features = [
        "top_left",
        "top_mid",
        "top_right",
        "center_left",
        "center_mid",
        "center_right",
        "bottom_left",
        "bottom_mid",
        "bottom_right",
    ]

    # concert tensor ds into numpy array
    images = np.concatenate([x for x, _ in ds], axis=0)

    # one-hot encoded labels
    labels = np.concatenate([y for _, y in ds], axis=0)

    # convert to proper labels
    labels = np.argmax(labels, axis=-1)

    # resize the image using opencv function
    # ravel the images into array with 9 features.
    resized = np.array([cv2.resize(img, (3, 3), interpolation=cv2.INTER_AREA).ravel() for img in images])

    # scale resize
    scaled_resize = get_scaled_dataset(resized)

    # convert this numpy array to panda dataframe
    # adapted from: https://datatofish.com/numpy-array-to-pandas-dataframe/
    df = pd.DataFrame(
        scaled_resize,
        columns=features,
    )

    aucs = pd.DataFrame(
        columns=cols,
        data=np.zeros([len(features), len(cols)]),
    )

    # multiclass classification
    for i, feature in enumerate(features):
        auc = roc_auc_score(y_true=labels, y_score=df[feature], multi_class="ovr")
        print(auc)
        auc.iloc[i] = (feature, auc)

    # this code I adapted from my assignment2 submission
    # sorting, with key function, reference: official documentation
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sort_values.html
    # sorts aucs, farthest from 0.5 in descending order
    aucs_sorted = aucs.sort_values(by=["AUC"], key=lambda e: abs(e - 0.5), ascending=False)

    # save to proper directory, # fig
    dir_path = Path(__file__).parent.parent.joinpath("/outputs/").resolve()
    aucs_sorted.to_json(f"{dir_path} / 'aucs.json'")
    print(aucs_sorted[:10].round(3).to_markdown(index=False))
    return None
