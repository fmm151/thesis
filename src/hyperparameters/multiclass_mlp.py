# Help from: https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python
# Code from Explainable AI with Python

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.src.utils import np_utils
from sklearn.model_selection import train_test_split

tf.compat.v1.disable_v2_behavior()

from keras import callbacks
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

samples_number = 100
number_of_classes = 94

# Dataset to load
filename = "../../files/labeled_datasets_features/labeled_dataset_features.csv"


def load_dataset(filename):
    # Load the dataset in the form of a csv
    df = pd.read_csv(filename)
    headers = pd.read_csv(filename, index_col=False, nrows=0).columns.tolist()
    features = headers[0:-3]

    # Return a dataframe and the names of the features
    return df, features


def drop_features_by_correlation(df):
    # Calculate correlation coefficients
    df_for_corr = df.drop(labels=['Name', 'Label', 'Family'], axis=1)
    correlation_coeffs = df_for_corr.corr()

    # Keep the upper triangular matrix of correlation coefficients
    upper_tri = correlation_coeffs.where(np.triu(np.ones(correlation_coeffs.shape), k=1).astype(np.bool_))

    # Drop columns with high correlation
    to_drop = [column for column in upper_tri.columns if any(abs(upper_tri[column]) >= 0.9)]
    df = df.drop(columns=to_drop, inplace=False)
    features = df.columns.tolist()[0:-3]

    # Return dropped features and the new dataframe
    return to_drop, df, features


def split_dataset(df):
    # Split the dataset into training and testing sets
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=2345, shuffle=True)

    # Split features from labels
    X_train = train_set.iloc[:, :-3]
    y_train = train_set.iloc[:, -1:]
    X_test = test_set.iloc[:, :-3]
    y_test = test_set.iloc[:, -3:]

    return X_train, y_train, X_test, y_test


def scale_dataset(X_train, X_test):
    # Scale the dataset using min max scaling
    minimum = X_train.min()
    maximum = X_train.max()
    X_train = (X_train - minimum) / (maximum - minimum)
    X_test = (X_test - minimum) / (maximum - minimum)

    return X_train, X_test


def oversample_data(X_train, y_train):
    # Oversample the data using SMOTE
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    return X_train, y_train


def train_model(X_train, y_train, algorithm, X_test, y_test_temp):
    rng = np.random.RandomState(42)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    open("../../files/results/predictions_multiclass_mlp.csv", "w").close()

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test_temp = label_encoder.fit_transform(y_test_temp)

    print("Y TRAIN", y_train)
    print("Y test temp", y_test_temp)

    y_train = np_utils.to_categorical(y_train, num_classes=number_of_classes)
    y_test_temp = np_utils.to_categorical(y_test_temp, num_classes=number_of_classes)

    print("Y TRAIN", y_train)
    print("Y test temp", y_test_temp)

    # for i in range(100, 310, 100):
    #     model_gs = tf.keras.models.Sequential()
    #     features_number = len(X_train.columns)
    #     model_gs.add(tf.keras.layers.Dense(i, input_dim = features_number, activation = 'relu'))
    #     model_gs.add(tf.keras.layers.Dropout(0.2))
    #
    #     model_gs.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))
    #
    #     model_gs.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    #     history = model_gs.fit(X_train, y_train, validation_split = 0.2, epochs = 100, batch_size = 256, callbacks = [early_stopping])
    #     print("batch: 256")
    #     print("dropout: 0.2")
    #     print(model_gs.summary())
    #     score = model_gs.evaluate(X_test, y_test_temp.values, verbose = 1)
    #     print(score[0])
    #     print(score[1])

    #!!!! for i in range(100, 310, 100):
    #     model_gs = tf.keras.models.Sequential()
    #     features_number = len(X_train.columns)
    #     model_gs.add(tf.keras.layers.Dense(i, input_dim=features_number, activation='relu'))
    #     model_gs.add(tf.keras.layers.Dropout(0.5))
    #
    #     # Change the output layer for multiclass classification
    #     num_classes = number_of_classes  # Replace with the actual number of classes in your problem
    #     model_gs.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    #
    #     model_gs.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #     history = model_gs.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=256,
    #                            callbacks=[early_stopping])
    #     print("batch: 256")
    #     print("dropout: 0.5")
    #     print(model_gs.summary())
    #
    #     predictions = model_gs.predict(X_test)
    #     print(predictions)
    #     print((y_test_temp))
    #     to_write = ", ".join(str(predictions))
    #     to_write_ = ", ".join(str(y_test_temp))
    #     with open('../../files/results/predictions_multiclass_mlp.csv', 'a') as f:
    #         f.write(to_write + "\n" + to_write_ + "\n" + "\n")
    #
    #     score = model_gs.evaluate(X_test, y_test_temp, verbose=1)
    #     print(score[0])
    #     print(score[1])

    for i in range(100, 310, 100):
        model_gs = tf.keras.models.Sequential()
        features_number = len(X_train.columns)
        model_gs.add(tf.keras.layers.Dense(i, input_dim=features_number, activation='relu'))
        model_gs.add(tf.keras.layers.Dropout(0.2))

        # Change the output layer for multiclass classification
        num_classes = number_of_classes
        model_gs.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

        model_gs.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model_gs.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=512,
                               callbacks=[early_stopping])
        print("batch: 512")
        print("dropout: 0.2")
        print(model_gs.summary())

        score = model_gs.evaluate(X_test, y_test_temp, verbose=1)
        print(score[0])
        print(score[1])

    # for i in range(100, 310, 100):
    #     model_gs = tf.keras.models.Sequential()
    #     features_number = len(X_train.columns)
    #     model_gs.add(tf.keras.layers.Dense(i, input_dim = features_number, activation = 'relu'))
    #     model_gs.add(tf.keras.layers.Dropout(0.5))
    #
    #     # Change the output layer for multiclass classification
    #     num_classes = number_of_classes  # Replace with the actual number of classes in your problem
    #     model_gs.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    #
    #     model_gs.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #     history = model_gs.fit(X_train, y_train, validation_split = 0.2, epochs = 100, batch_size = 512, callbacks = [early_stopping])
    #     print("batch: 512")
    #     print("dropout: 0.5")
    #     print(model_gs.summary())
    #     score = model_gs.evaluate(X_test, y_test_temp.values, verbose = 1)
    #     print(score[0])
    #     print(score[1])

    # for i in range(100, 310, 100):
    #     for j in range(100, 310, 100):
    #         model_gs = tf.keras.models.Sequential()
    #         features_number = len(X_train.columns)
    #         model_gs.add(tf.keras.layers.Dense(i, input_dim = features_number, activation = 'relu'))
    #         model_gs.add(tf.keras.layers.Dropout(0.2))
    #         model_gs.add(tf.keras.layers.Dense(j, activation = 'relu'))
    #         model_gs.add(tf.keras.layers.Dropout(0.2))
    #
    #         # Change the output layer for multiclass classification
    #         num_classes = number_of_classes  # Replace with the actual number of classes in your problem
    #         model_gs.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    #
    #         model_gs.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #         history = model_gs.fit(X_train, y_train, validation_split = 0.2, epochs = 100, batch_size = 256, callbacks = [early_stopping])
    #         print("batch: 256")
    #         print("dropout: 0.2")
    #         print(model_gs.summary())
    #
    #         score = model_gs.evaluate(X_test, y_test_temp.values, verbose = 1)
    #         print(score[0])
    #         print(score[1])

    # for i in range(100, 310, 100):
    #     for j in range(100, 310, 100):
    #         model_gs = tf.keras.models.Sequential()
    #         features_number = len(X_train.columns)
    #         model_gs.add(tf.keras.layers.Dense(i, input_dim = features_number, activation = 'relu'))
    #         model_gs.add(tf.keras.layers.Dropout(0.5))
    #         model_gs.add(tf.keras.layers.Dense(j, activation = 'relu'))
    #         model_gs.add(tf.keras.layers.Dropout(0.5))
    #
    #         # Change the output layer for multiclass classification
    #         num_classes = number_of_classes  # Replace with the actual number of classes in your problem
    #         model_gs.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    #
    #         model_gs.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #         history = model_gs.fit(X_train, y_train, validation_split = 0.2, epochs = 100, batch_size = 256, callbacks = [early_stopping])
    #         print("batch: 256")
    #         print("dropout: 0.5")
    #         print(model_gs.summary())
    #
    #         score = model_gs.evaluate(X_test, y_test_temp.values, verbose = 1)
    #         print(score[0])
    #         print(score[1])

    # for i in range(100, 310, 100):
    #     for j in range(100, 310, 100):
    #         model_gs = tf.keras.models.Sequential()
    #         features_number = len(X_train.columns)
    #         model_gs.add(tf.keras.layers.Dense(i, input_dim = features_number, activation = 'relu'))
    #         model_gs.add(tf.keras.layers.Dropout(0.2))
    #         model_gs.add(tf.keras.layers.Dense(j, activation = 'relu'))
    #         model_gs.add(tf.keras.layers.Dropout(0.2))
    #
    #         # Change the output layer for multiclass classification
    #         num_classes = number_of_classes  # Replace with the actual number of classes in your problem
    #         model_gs.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    #
    #         model_gs.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #         history = model_gs.fit(X_train, y_train, validation_split = 0.2, epochs = 100, batch_size = 512, callbacks = [early_stopping])
    #         print("batch: 512")
    #         print("dropout: 0.2")
    #         print(model_gs.summary())
    #
    #         score = model_gs.evaluate(X_test, y_test_temp.values, verbose = 1)
    #         print(score[0])
    #         print(score[1])

    #!!!!! for i in range(100, 310, 100):
    #     for j in range(100, 310, 100):
    #         model_gs = tf.keras.models.Sequential()
    #         features_number = len(X_train.columns)
    #         model_gs.add(tf.keras.layers.Dense(i, input_dim=features_number, activation='relu'))
    #         model_gs.add(tf.keras.layers.Dropout(0.5))
    #         model_gs.add(tf.keras.layers.Dense(j, activation='relu'))
    #         model_gs.add(tf.keras.layers.Dropout(0.5))
    #
    #         # Change the output layer for multiclass classification
    #         num_classes = number_of_classes  # Replace with the actual number of classes in your problem
    #         model_gs.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    #
    #         model_gs.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #         history = model_gs.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=512,
    #                                callbacks=[early_stopping])
    #         print("batch: 512")
    #         print("dropout: 0.5")
    #         print(model_gs.summary())
    #
    #         predictions = model_gs.predict(X_test)
    #         to_write = ", ".join(predictions)
    #         to_write_ = ", ".join(y_test_temp.values)
    #         with open('../../files/results/predictions_multiclass_mlp.csv', 'a') as f:
    #             f.write(to_write + "\n" + to_write_ + "\n" + "\n")
    #
    #         score = model_gs.evaluate(X_test, y_test_temp.values, verbose=1)
    #         print(score[0])
    #         print(score[1])

    # for i in range(100, 310, 100):
    #     for j in range(100, 310, 100):
    #         for k in range(100, 310, 100):
    #             model_gs = tf.keras.models.Sequential()
    #             features_number = len(X_train.columns)
    #             model_gs.add(tf.keras.layers.Dense(i, input_dim = features_number, activation = 'relu'))
    #             model_gs.add(tf.keras.layers.Dropout(0.2))
    #             model_gs.add(tf.keras.layers.Dense(j, activation = 'relu'))
    #             model_gs.add(tf.keras.layers.Dropout(0.2))
    #             model_gs.add(tf.keras.layers.Dense(k, activation = 'relu'))
    #             model_gs.add(tf.keras.layers.Dropout(0.2))
    #
    #             # Change the output layer for multiclass classification
    #             num_classes = number_of_classes  # Replace with the actual number of classes in your problem
    #             model_gs.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    #
    #             model_gs.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    #             history = model_gs.fit(X_train, y_train, validation_split = 0.2, epochs = 100, batch_size = 256, callbacks = [early_stopping])
    #             print("batch: 256")
    #             print("dropout: 0.2")
    #             print(model_gs.summary())
    #
    #             score = model_gs.evaluate(X_test, y_test_temp, verbose = 1)
    #             print(score[0])
    #             print(score[1])

    # for i in range(100, 310, 100):
    #     for j in range(100, 310, 100):
    #         for k in range(100, 310, 100):
    #             model_gs = tf.keras.models.Sequential()
    #             features_number = len(X_train.columns)
    #             model_gs.add(tf.keras.layers.Dense(i, input_dim = features_number, activation = 'relu'))
    #             model_gs.add(tf.keras.layers.Dropout(0.5))
    #             model_gs.add(tf.keras.layers.Dense(j, activation = 'relu'))
    #             model_gs.add(tf.keras.layers.Dropout(0.5))
    #             model_gs.add(tf.keras.layers.Dense(k, activation = 'relu'))
    #             model_gs.add(tf.keras.layers.Dropout(0.5))
    #
    #             # Change the output layer for multiclass classification
    #             num_classes = number_of_classes  # Replace with the actual number of classes in your problem
    #             model_gs.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    #
    #             model_gs.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    #             history = model_gs.fit(X_train, y_train, validation_split = 0.2, epochs = 100, batch_size = 256, callbacks = [early_stopping])
    #             print("batch: 256")
    #             print("dropout: 0.5")
    #             print(model_gs.summary())
    #
    #             score = model_gs.evaluate(X_test, y_test_temp.values, verbose = 1)
    #             print(score[0])
    #             print(score[1])

    # for i in range(100, 310, 100):
    #     for j in range(100, 310, 100):
    #         for k in range(100, 310, 100):
    #             model_gs = tf.keras.models.Sequential()
    #             features_number = len(X_train.columns)
    #             model_gs.add(tf.keras.layers.Dense(i, input_dim = features_number, activation = 'relu'))
    #             model_gs.add(tf.keras.layers.Dropout(0.2))
    #             model_gs.add(tf.keras.layers.Dense(j, activation = 'relu'))
    #             model_gs.add(tf.keras.layers.Dropout(0.2))
    #             model_gs.add(tf.keras.layers.Dense(k, activation = 'relu'))
    #             model_gs.add(tf.keras.layers.Dropout(0.2))
    #
    #             # Change the output layer for multiclass classification
    #             num_classes = number_of_classes  # Replace with the actual number of classes in your problem
    #             model_gs.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    #
    #             model_gs.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    #             history = model_gs.fit(X_train, y_train, validation_split = 0.2, epochs = 100, batch_size = 512, callbacks = [early_stopping])
    #             print("batch: 512")
    #             print("dropout: 0.2")
    #             print(model_gs.summary())
    #
    #             score = model_gs.evaluate(X_test, y_test_temp.values, verbose = 1)
    #             print(score[0])
    #             print(score[1])

    #!!!!!! for i in range(100, 310, 100):
    #     for j in range(100, 310, 100):
    #         for k in range(100, 310, 100):
    #             model_gs = tf.keras.models.Sequential()
    #             features_number = len(X_train.columns)
    #             model_gs.add(tf.keras.layers.Dense(i, input_dim=features_number, activation='relu'))
    #             model_gs.add(tf.keras.layers.Dropout(0.5))
    #             model_gs.add(tf.keras.layers.Dense(j, activation='relu'))
    #             model_gs.add(tf.keras.layers.Dropout(0.5))
    #             model_gs.add(tf.keras.layers.Dense(k, activation='relu'))
    #             model_gs.add(tf.keras.layers.Dropout(0.5))
    #
    #             # Change the output layer for multiclass classification
    #             num_classes = number_of_classes  # Replace with the actual number of classes in your problem
    #             model_gs.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    #
    #             model_gs.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #             history = model_gs.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=512,
    #                                    callbacks=[early_stopping])
    #             print("batch: 512")
    #             print("dropout: 0.5")
    #             print(model_gs.summary())
    #
    #             predictions = model_gs.predict(X_test)
    #             to_write = ", ".join(predictions)
    #             to_write_ = ", ".join(y_test_temp.values)
    #             with open('../../files/results/predictions_multiclass_mlp.csv', 'a') as f:
    #                 f.write(to_write + "\n" + to_write_ + "\n" + "\n")
    #
    #             score = model_gs.evaluate(X_test, y_test_temp.values, verbose=1)
    #             print(score[0])
    #             print(score[1])
    #
    # return model_gs


if __name__ == "__main__":
    df, features = load_dataset(filename)

    to_drop, df, features = drop_features_by_correlation(df)
    print("Dropped features: ", str(to_drop))

    X_train, y_train, X_test, y_test = split_dataset(df)
    X_train, X_test = scale_dataset(X_train, X_test)
    # X_train, y_train = oversample_data(X_train, y_train)

    algorithm = "mlp"
    y_test_temp = y_test.iloc[:, 2]
    print("Y TEST TEMP")
    print(y_test_temp)
    model_gs = train_model(X_train, y_train, algorithm, X_test, y_test_temp)
