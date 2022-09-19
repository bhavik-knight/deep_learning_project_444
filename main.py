###############################################################################
# Name:             Bhavik Bhagat
# Email:            x2020coq@stfx.ca
# Student ID:       202002911
# Course:           csci444, final project
###############################################################################

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.helpers import plot, plot_confusion_matrix, read_data
from src.models import get_cnn


def main():
    # get the data for CNN
    # this have several parameters that can be tweaked
    # some parts are adapted from this
    # https://www.youtube.com/playlist?list=PLZbbT5o_s2xrwRnXk_yCPtnqqo4_u2YGLrts

    train, test, val, pred = read_data(model="cnn", augment=False, prefetch=False)

    # get the model
    for cnn in get_cnn().values():

        # save the architecture of the model
        dir_path = Path(__file__).parent.joinpath("outputs").resolve()
        filename = dir_path.joinpath("cnn_arch.json")
        cnn_json = cnn.to_json()
        with open(filename, "w") as json_file:
            json_file.write(cnn_json)

        print("\nTraining started\n")
        history = cnn.fit(
            train, validation_data=val, batch_size=64, epochs=30, use_multiprocessing=True, workers=8, verbose=1
        )
        print("\nTraining completed\n")

        # now that training is complete, saving the model with learned weights
        # so that I can use this model on some other dataset
        dir_path = Path(__file__).parent.joinpath("/outputs/").resolve()
        filename = dir_path.joinpath("cnn_intel.h5")
        if os.path.isfile(filename) is False:
            cnn.save(filename)

        print("\nTesting started\n")
        cnn.evaluate(test, batch_size=64, use_multiprocessing=True, workers=8, verbose=1, return_dict=True)
        print("\nTesting completed\n")

        # plot the results
        plot("cnn", history.history, validation=False)
        plot("cnn_val", history.history, validation=True)

        # for confusion matrix
        predictions = cnn.predict(test, batch_size=64, use_multiprocessing=True, workers=8, verbose=1)

        images = np.concatenate([x for x, _ in test], axis=0)
        predictions = np.argmax(predictions, axis=-1)
        labels = np.array([])
        for x, y in test:
            labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])
        cm = tf.math.confusion_matrix(labels, predictions).numpy()
        plot_confusion_matrix(cm, classes=labels)

        # try to plot some correct predictions, from the test set
        predicted_classes = np.argmax(np.round(predictions), axis=-1)
        correct = np.where(predicted_classes == labels)[0]
        # print some correct stuff
        for i, correct in enumerate(correct[:9]):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[correct], cmap="gray", interpolation="none")
            plt.title("Predicted {}, Class {}".format(predicted_classes[correct], labels[correct]))
            plt.tight_layout()

            dir_path = Path(__file__).parent.joinpath("/figs/").resolve()
            filename = dir_path.joinpath("predictions.png")
            plt.savefig(filename)
            # plt.show()

    # for knn, svm, rf, I converted color images to grayscale to reduce the run time
    # source: https://www.tensorflow.org/api_docs/python/tf/image/rgb_to_grayscale
    # TODO KNN, SVM, RF
    return None


def set_GPU():
    # my GPU is causing so many errors, it's indeed a headache
    # https://tensorflow.google.cn/guide/gpu?hl=en
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Restrict TensorFlow to only allocate 1.7GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1700)]
            )

            tf.config.experimental.set_virtual_device_configuration(
                gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1700)]
            )

            logical_gpus = tf.config.list_logical_devices("GPU")
            # tf.config.experimental.set_memory_growth(gpus[0], True)
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)

    os.environ["TF_FOURCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
    os.environ["TF_CPP_VMODULE"] = "gpu_process_state=20, gpu_cudamallocasync_allocator=20"

    print("\nGPU is all set now.\n")
    return None


def turn_off_GPU():
    # turning off GPU because, I am lagging in actual project because of this
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if tf.test.gpu_device_name():
        print("GPU Found")
    else:
        print("NO GPU Found")
    return None


if __name__ == "__main__":
    set_GPU()  # please use this, if you get any GPU memory related errors
    # turn_off_GPU()          # please turn GPU off is you cannot make GPU work
    main()
