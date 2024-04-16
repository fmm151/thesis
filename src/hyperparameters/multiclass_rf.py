# Help from: https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python
# Code from Explainable AI with Python

# True to print debugging outputs, False to silence the program
DEBUG = True
separator = "-------------------------------------------------------------------------"
# Correlation threshold for Pearson correlation. For feature pairs with correlation higher than the threshold, one feature is dropped
CORRELATION_THRESHOLD = 0.9

# Import the necessary libraries (tested for Python 3.9)
import numpy as np
import os
import pandas as pd
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import sys

# Dataset to load
filename = "../../files/labeled_datasets_features/multiclass/multiclass_features_10K.csv"

# file_out = "../../files/results/grid_search_results_multiclass_rf.csv"
# fdw = open(file_out, "w")
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

families_mapping = {family: index for index, family in enumerate(families)}

def load_dataset(filename, families):
    # Load the dataset in the form of a csv
    df = pd.read_csv(filename)
    headers = pd.read_csv(filename, index_col=False, nrows=0).columns.tolist()
    features = headers[0:-3]

    family_counts = df['Family'].value_counts()
    print("The number of samples for each family: \n", family_counts)

    # Filter classes with counts greater than or equal to 100
    valid_classes_500 = family_counts[family_counts >= 500].index
    valid_classes_1000 = family_counts[family_counts >= 1000].index
    valid_classes_2000 = family_counts[family_counts >= 2000].index
    families = [item for item in families if item in valid_classes_1000]
    print("NOW HE HAVE", len(families), "families")

    # Filter the DataFrame to keep only rows with valid classes
    df = df[df['Family'].isin(valid_classes_1000)]

    family_counts = df['Family'].value_counts()
    print("The number of samples for each family:", family_counts)

    # Return a dataframe and the names of the features
    return df, features, families


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
    y_train = train_set.iloc[:, -1:]
    X_test = test_set.iloc[:, :-3]
    y_test = test_set.iloc[:, -3:]

    print("Y TRAIN!!!!!", y_train)
    print("Y TEST!!!!!!", y_test.iloc[:, -1:])
    print("The mapping:\n", families_mapping)
    y_train_original = y_train
    # label_encoder = LabelEncoder()
    # y_train = label_encoder.fit_transform(y_train.values.ravel())
    y_train = y_train_original.replace(families_mapping)

    # # Extract the mapping between original labels and their encoded values
    # label_mapping = {original_label: encoded_label for original_label, encoded_label in
    #                  zip(y_train_original.values.ravel(), y_train)}

    # y_train = pd.DataFrame(y_train, columns=['Family'])
    # y_test.iloc[:, -1:] = label_encoder.fit_transform(y_test.iloc[:, -1:].values.ravel())
    y_test.iloc[:,-1:] = y_test.iloc[:,-1:].replace(families_mapping)

    print("Y TRAIN!!!!!", y_train)
    print("Y TEST!!!!!!", y_test.iloc[:, -1:])

    return X_train, y_train, X_test, y_test #, label_mapping


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

    print("COUNT!!!!!!!!!!!!!", y_train.value_counts())
    X_train, y_train = sm.fit_resample(X_train, y_train)

    return X_train, y_train

def train_and_test_model(X_train, y_train, X_test, y_test):
    rng = np.random.RandomState(42)

    # # os.remove("../../files/results/results_multiclass_rf.csv")
    # open("../../files/results/results_multiclass_rf.csv", "w").close()
    # open("../../files/results/predictions_multiclass_rf.csv", "w").close()

    for estimators in range(10, 250, 50):
        for depth in range(10, 250, 50):
            model = RandomForestClassifier(n_estimators=estimators, max_depth=depth, n_jobs=-1, random_state=rng)
            model.fit(X_train, y_train.values.ravel())
            predictions = model.predict(X_test)

            arr1 = predictions.astype(int)
            arr2 = y_test["Family"].values.astype(int)
            accuracy = accuracy_score(arr2, arr1, normalize=True)

            if DEBUG == True:
                txt = str(accuracy) + "," + str(estimators) + "," + str(depth)
                print(txt)

            # to_write = str(accuracy) + "," + str(estimators) + "," + str(depth)
            # with open('../../files/results/results_multiclass_rf_30Κ.csv', 'a') as f:
            #     f.write(to_write + "\n")

            # to_write = ", ".join(predictions)
            # to_write_ = ", ".join(y_test["Family"].values)
            # with open('../../files/results/predictions_multiclass_rf.csv', 'a') as f:
            #     f.write(to_write + "\n" + to_write_ + "\n" + "\n")
            # # fdw.write(to_write + "\n")

    return None


if __name__ == "__main__":
    # Load the dataset
    df, features, families = load_dataset(filename, families)

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