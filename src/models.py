from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.utils import plot_model

# I know the shape of images in 150x150
# I am hard-coding this here, because I am loading dataset in main file
# loading at multiple places is inefficient
shape = (150, 150, 1)
num_classes = 6


# gather the different kinds of models
def get_all_models():
    models = dict()
    # put all this models in dictionary
    models.update(get_knn())
    models.update(get_svm())
    models.update(get_rf())
    models.update(get_ann())
    models.update(get_cnn())

    return models.copy()


# KNNs, with k = 10, 20 neighbors
def get_knn():
    knn10 = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
    knn20 = KNeighborsClassifier(n_neighbors=20, n_jobs=-1)
    return {"knn10": knn10, "knn20": knn20}


# SVMs
def get_svm():
    svm_linear = SVC(kernel="linear")
    svm_rbf = SVC(kernel="rbf")
    return {"svm_linear": svm_linear, "svm_rbf": svm_rbf}


# RF
def get_rf():
    return {"rf": RandomForestClassifier(n_estimators=100)}


# ANNs, ann10, ann50 uses Scikit Learn, ann64 uses Keras
def get_ann():
    # use scikit learn MLP for ann with 10, 50 hidden layers
    ann10 = MLPClassifier(
        hidden_layer_sizes=10, activation="relu", solver="adam", alpha=0.0001, batch_size=64, max_iter=100
    )

    ann50 = MLPClassifier(
        hidden_layer_sizes=50, activation="relu", solver="adamax", alpha=0.0001, batch_size=64, max_iter=100
    )

    # user tensorflow, keras to build ANN
    ann = Sequential([Dense(units=num_classes, use_bias=True, activation="softmax")])
    ann.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")

    return {"ann10": ann10, "ann50": ann50, "ann": ann}


# CNN
# started with LeNet, but improvement on it
def get_cnn():
    kernel_size = 5
    cnn = Sequential(
        [
            # 1. conv2D to generate 32 feature maps, with 5x5 kernel, unit stride, and relu activation
            Conv2D(
                filters=32,
                kernel_size=kernel_size,
                input_shape=shape,
                activation="relu",
            ),
            # 2. conv2D to generate 32 feature maps, with 5x5 kernel, using bias this time
            Conv2D(filters=32, kernel_size=kernel_size, use_bias=True),
            # 3. normalizing
            BatchNormalization(),
            # 4. pooling layer, to pull the best in small grid, using max pool, with 2x2 pool_size (kernel)
            Activation("relu"),
            MaxPooling2D(pool_size=2, strides=2),
            # 5. dropping some learned parameters, to avoid memorization (over-fitting) instead of learning
            Dropout(0.30),
            # 6 - 10: do all of steps 1-5 with slightly different parameters
            Conv2D(filters=64, kernel_size=3, activation="relu"),
            Conv2D(filters=64, kernel_size=3, use_bias=True),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(pool_size=2, strides=2),
            Dropout(0.20),
            # # 11, 12: flattening it then making it fully connected, using Dense
            # # keep normalizing to make image stable, (vary less) and
            Flatten(),
            Dense(units=256, use_bias=True),
            BatchNormalization(),
            # 13, 14: activate, another fully connected layer, with bias
            Dense(units=64, use_bias=True, activation="relu"),
            BatchNormalization(),
            Dropout(0.25),
            # 15, 16: do the previous step one more time, we have multi-class problem w/ 6 classes
            Dense(units=num_classes, use_bias=False, activation="softmax"),
        ]
    )
    # cnn building layers is complete, show summary
    # cnn.summary()

    # now compile that model
    cnn.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")

    # save this model in the correct directory
    dir_path = Path(__file__).parent.parent.joinpath("figs/models").resolve()
    file_path = Path(dir_path).joinpath("model_cnn.png")
    plot_model(cnn, to_file=file_path, show_shapes=True)

    return {"cnn": cnn}
