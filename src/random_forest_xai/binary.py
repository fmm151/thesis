# Help from: https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python
# Code from Explainable AI with Python

# True to print debugging outputs, False to silence the program
DEBUG = True
separator = "-------------------------------------------------------------------------"
# Define the number of clusters that will represent the training dataset for SHAP framework (cannot give all training samples)
K_MEANS_CLUSTERS = 100
# Define the number of testing samples on which SHAP will derive interpretations
SAMPLES_NUMBER = 300
# Correlation threshold for Pearson correlation. For feature pairs with correlation higher than the threshold, one feature is dropped
CORRELATION_THRESHOLD = 0.9

# Import the necessary libraries (tested for Python 3.9)
import os
import numpy as np
import pandas as pd
from sklearn import ensemble
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Input, Dropout
from keras import callbacks
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import eli5
from eli5.sklearn import PermutationImportance
from pdpbox import pdp, info_plots
import shap
import sys

# Dataset to load
filename = "../../files/labeled_dataset_features.csv"

# Families to be more used
thesis_families = ["bamital", "dircrypt", "matsnu", "volatilecedar", "all_DGAs"]

# Families considered for SHAP interpretations
families = ["tranco", "bamital", "banjori", "bedep", "beebone", "blackhole", "bobax", "ccleaner",
        "chinad", "chir", "conficker", "corebot", "cryptolocker", "darkshell", "diamondfox", "dircrypt",
        "dmsniff", "dnsbenchmark", "dnschanger", "downloader", "dyre", "ebury", "ekforward", "emotet",
        "feodo", "fobber", "gameover", "gozi", "goznym", "gspy", "hesperbot", "infy",
        "locky", "madmax", "makloader", "matsnu", "mirai", "modpack", "monerominer", "murofet",
        "murofetweekly", "mydoom", "necurs", "nymaim2", "nymaim", "oderoor", "omexo", "padcrypt",
        "pandabanker", "pitou", "proslikefan", "pushdo", "pushdotid", "pykspa2", "pykspa2s", "pykspa",
        "qadars", "qakbot", "qhost", "qsnatch", "ramdo", "ramnit", "ranbyus", "randomloader", "redyms", "rovnix",
        "shifu", "simda", "sisron", "sphinx", "suppobox", "sutra", "symmi", "szribi", "tempedreve",
        "tempedrevetdd", "tinba", "tinynuke", "tofsee", "torpig", "tsifiri", "ud2", "ud3", "ud4", "urlzone", "vawtrak",
        "vidro", "vidrotid", "virut", "volatilecedar", "wd", "xshellghost", "xxhex"]



def load_dataset(filename):
    # Load the dataset in the form of a csv
    df = pd.read_csv(filename)
    headers = pd.read_csv(filename, index_col=False, nrows=0).columns.tolist()
    features = headers[0:-3]

    # Return a dataframe and the names of the features
    return df, features


def drop_features_by_correlation(df):
    # Calculate correlation coefficients for pairs of features
    df_for_corr = df.drop(labels=['Name', 'Label', 'Family'], axis=1)
    correlation_coeffs = df_for_corr.corr()

    # Keep the upper triangular matrix of correlation coefficients
    upper_tri = correlation_coeffs.where(np.triu(np.ones(correlation_coeffs.shape), k=1).astype(np.bool_))

    # Drop columns with high correlation (one of the features consisting the pair is dropped, the other is kept)
    to_drop = [column for column in upper_tri.columns if any(abs(upper_tri[column]) >= CORRELATION_THRESHOLD)]

    if DEBUG == True:
        print("Correlation threshold is:")
        print(CORRELATION_THRESHOLD)
        print(separator)

    df = df.drop(columns=to_drop, inplace=False)
    features = df.columns.tolist()[0:-3]

    # Return the names of the dropped features, the new dataframe and the names of the new features within the feature set
    return to_drop, df, features


def split_dataset(df):
    # Split the dataset into training and testing sets
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=2345, shuffle=True)

    # Split features from labels (the last three columns are domain name, binary label, malware family)
    X_train = train_set.iloc[:, :-3]
    y_train = train_set.iloc[:, -2:-1]
    X_test = test_set.iloc[:, :-3]
    y_test = test_set.iloc[:, -3:]


    return X_train, y_train, X_test, y_test


def scale_dataset(X_train, X_test):
    # Scale the dataset using min-max scaling
    minimum = X_train.min()
    maximum = X_train.max()
    X_train = (X_train - minimum) / (maximum - minimum)
    X_test = (X_test - minimum) / (maximum - minimum)

    # Return the scaled training and testing datasets
    return X_train, X_test


def oversample_data(X_train, y_train):
    # Oversample the data using SMOTE
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    return X_train, y_train


def train_and_test_model(X_train, y_train, X_test, y_test):
    rng = np.random.RandomState(42)

    model = RandomForestClassifier(n_estimators=10, max_depth=50, n_jobs=-1, random_state=rng)
    model.fit(X_train, y_train.values.ravel())
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test["Label"].values, predictions, normalize=True)

    # Print the different testing scores
    print("Algorithm: Random Forest")
    print("Accuracy: ", accuracy_score(y_test["Label"].values, predictions, normalize=True))
    print("Precision None: ", precision_score(y_test["Label"].values, predictions, average=None))
    print("Recall None: ", recall_score(y_test["Label"].values, predictions, average=None))
    print("F1 score None: ", f1_score(y_test["Label"].values, predictions, average=None))
    print(separator)

    # print("Accuracy score for 10 estimators and max depth 50: ", accuracy)

    return None

def evaluate_model_on_family(model, family, x_testing, algorithm):
    # Calculate accuracy for a specific malware family
    predictions = model.predict(x_testing)
    non_zero_values = np.count_nonzero(predictions)
    if family != "tranco":
        accuracy = non_zero_values / predictions.size
    else:
        zero_values = predictions.size - non_zero_values
        accuracy = zero_values / predictions.size
    print("Accuracy on sampled testing dataset for family and algorithm: ", family, algorithm, accuracy)
    return None

def feature_importance_using_permutation_importance(model, X_test, y_test, features, algorithm):
    # Use permutation importance to evaluate the importance of the features
    # Currently supported only for Random Forests
    perm = PermutationImportance(model, random_state=1).fit(X_test, y_test["Label"])
    result = eli5.format_as_text(eli5.explain_weights(perm, top=100, feature_names=features))
    print("Permutation importance results for algorithm: ", algorithm)
    print(result)
    print(separator)
    return None

def split_testing_dataset_into_categories(X_test, y_test):
    # For SHAP purposes, to derive interpretations per malware family, we will split the testing dataset per malware family
    test_merged = pd.merge(left=X_test, left_index=True, right=y_test, right_index=True, how='inner')

    per_category_test = {}

    # For all DGA's regardless of the malware family
    all_test = test_merged.iloc[:, :-2]
    per_category_test["all"] = all_test

    per_category_test["all_DGAs"] = test_merged[test_merged.iloc[:, -1] != "tranco"].iloc[:, :-2]

    for family in families:
        test = test_merged[test_merged.iloc[:, -1] == str(family)]
        test = test.iloc[:, :-2]
        if len(test) > 0:
            per_category_test[str(family)] = test

    # This will hold testing samples per malware family, e.g. per_category_test["bamital"] holds testing samples for bamital
    return per_category_test


if __name__ == "__main__":
    # Load the dataset
    df, features = load_dataset(filename)

    if DEBUG == True:
        print("Before correlation: The dataframe is:")
        print(df)
        print(separator)
        print("Before correlation: The shape of the dataframe is:")
        print(df.shape)
        print(separator)
        print("Before correlation: The names of the features are:")
        print(features)
        print(separator)

    # Drop features based on Pearson correlation
    to_drop, df, features = drop_features_by_correlation(df)
    print("Dropped features because of correlation: ", str(to_drop))

    if DEBUG == True:
        print("After correlation: The new dataframe is:")
        print(df)
        print(separator)
        print("After correlation: The shape of the dataframe is:")
        print(df.shape)
        print(separator)
        print("After correlation: The names of the features are:")
        print(features)
        print(separator)

    # Split dataset into training and testing portions
    X_train, y_train, X_test, y_test = split_dataset(df)

    if DEBUG == True:
        print("Unscaled X_train:")
        print(X_train)
        print(separator)
        print("Size of X_train:")
        print(len(X_train))
        print(separator)
        print("y_train:")
        print(y_train)
        print(separator)
        print("Size of y_train:")
        print(len(y_train))
        print(separator)
        print("Unscaled X_test:")
        print(X_test)
        print(separator)
        print("Size of X_test:")
        print(len(X_test))
        print(separator)
        print("y_test:")
        print(y_test)
        print(separator)
        print("Size of y_test:")
        print(len(y_test))
        print(separator)

    # Scale dataset using min-max scaling
    X_train, X_test = scale_dataset(X_train, X_test)

    if DEBUG == True:
        print("Scaled X_train:")
        print(X_train)
        print(separator)
        print("Scaled X_test:")
        print(X_test)
        print(separator)

    # Data oversampling to deal with class imbalance
    X_train, y_train = oversample_data(X_train, y_train)

    if DEBUG == True:
        print("Size of oversampled X_train:")
        print(len(X_train))
        print(separator)
        print("Size of oversampled y_train:")
        print(len(y_train))
        print(separator)

    train_and_test_model(X_train, y_train, X_test, y_test)