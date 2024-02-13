import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

tf.compat.v1.disable_v2_behavior()

from tensorflow.keras import callbacks
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


def train_model(X_train, y_train, algorithm, X_test, y_test_temp):
    rng = np.random.RandomState(42)
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test_temp = label_encoder.fit_transform(y_test_temp)

    print(y_train)
    print(y_test_temp)