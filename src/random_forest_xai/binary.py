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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import subprocess
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import shap

# Dataset to load
filename = "../../files/labeled_dataset_features.csv"

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


def train_model(X_train, y_train):
    rng = np.random.RandomState(42)

    model = RandomForestClassifier(n_estimators=10, max_depth=50, n_jobs=-1, random_state=rng)
    model.fit(X_train, y_train.values.ravel())

    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    # Print the different testing scores
    print("Algorithm: Random Forest")
    print("Accuracy: ", accuracy_score(y_test["Label"].values, predictions, normalize=True))
    print("Precision None: ", precision_score(y_test["Label"].values, predictions, average=None))
    print("Recall None: ", recall_score(y_test["Label"].values, predictions, average=None))
    print("F1 score None: ", f1_score(y_test["Label"].values, predictions, average=None))
    print(separator)

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

def explain_with_shap_summary_plots(model, model_shap_values, family, test_sample, algorithm):
    # Plot bar summary plot using SHAP values
    prepend_path = os.path.join("..", "..", "files", "results", str(algorithm), str(family), "summary-plots")
    command = "mkdir " + prepend_path
    subprocess.run(command, shell=True)

    fig = plt.clf()
    shap.summary_plot(model_shap_values, test_sample, plot_type="bar", show=False)
    name = os.path.join(prepend_path, f"{family}-summarybar-original-{algorithm}.png")
    plt.savefig(name)
    plt.close("all")

    # Plot summary plot using SHAP values
    fig = plt.clf()
    shap.summary_plot(model_shap_values, test_sample, show=False)
    name = os.path.join(prepend_path, f"{family}-summaryplot-original-{algorithm}.png")
    plt.savefig(name)
    plt.close("all")

    return None


def explain_with_shap_dependence_plots(model, model_shap_values, family, test_sample, features, algorithm):
    # Plot dependence plot using SHAP values for multiple features
    prepend_path = os.path.join("..", "..", "files", "results", str(algorithm), str(family), "dependence-plots")
    command = "mkdir " + prepend_path
    subprocess.run(command, shell=True)

    for feature in features:
        fig = plt.clf()
        shap.dependence_plot(feature, model_shap_values, test_sample, show=False)
        name = os.path.join(prepend_path, f"{family}-dependence-{feature}-original-{algorithm}.png")
        plt.savefig(name, bbox_inches='tight')
        plt.close("all")

    return None

def explain_with_force_plots(model, model_shap_values, family, test_sample, names_sample, algorithm, model_explainer):
    # Plot force plots using SHAP values (local explanations)
    prepend_path = os.path.join("..", "..", "files", "results", str(algorithm), str(family), "force-plots")
    command = "mkdir " + prepend_path
    subprocess.run(command, shell=True)

    predictions = model.predict(test_sample)
    index_values = list(test_sample.index.values)
    sequence = 0
    for index in index_values:
        original_name = names_sample[index]
        name = original_name.replace(".", "+")
        prediction = predictions[sequence]

        fig = plt.clf()
        shap.force_plot(model_explainer.expected_value, model_shap_values[sequence, :], test_sample.loc[index],
                        matplotlib=True, show=False)
        name_of_file = os.path.join(prepend_path, f"{family}-force-{sequence}-name-{name}-prediction-{prediction}-{algorithm}-original.png")
        plt.title(original_name, y=1.5)
        plt.savefig(name_of_file, bbox_inches='tight')
        plt.close("all")

        fig = plt.clf()
        shap.force_plot(model_explainer.expected_value, model_shap_values[sequence, :], test_sample.loc[index],
                        matplotlib=True, show=False, contribution_threshold=0.1)
        name_of_file = os.path.join(prepend_path, f"{family}-force-{sequence}-name-{name}-prediction-{prediction}-{algorithm}-threshold01.png")
        plt.title(original_name, y=1.5)
        plt.savefig(name_of_file, bbox_inches='tight')
        plt.close("all")

        sequence += 1
        # Plot only the first 100 or less if no more than 100 exist
        if sequence == 10:
            break

    # plt.clf()
    # shap.force_plot(
    #     model_explainer.expected_value,
    #     model_shap_values,
    #     matplotlib=True,
    #     show=False,
    # )
    #
    # name_of_combined_file = os.path.join(
    #     prepend_path, f"{family}-stacked-force-all-samples-{algorithm}.png"
    # )
    # plt.title("Combined Stacked Force Plot", y=1.5)
    # plt.savefig(name_of_combined_file, bbox_inches="tight")
    # plt.close("all")

    return None


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

    # Split testing dataset into categories based on malware family
    per_category_test = split_testing_dataset_into_categories(X_test, y_test)

    # Keeping the names will prove useful for local explainability (force plots)
    test_sample = {}
    names_sample = {}
    for family in per_category_test.keys():
        if DEBUG == True:
            print("Processing family: ", family)
        if len(per_category_test[family]) < SAMPLES_NUMBER:
            test_sample[family] = shap.utils.sample(per_category_test[family], len(per_category_test[family]),
                                                    random_state=1452)
        else:
            test_sample[family] = shap.utils.sample(per_category_test[family], SAMPLES_NUMBER, random_state=1452)
        names_sample[family] = test_sample[family].iloc[:, -1]
        test_sample[family] = test_sample[family].iloc[:, 0:-1]

    if DEBUG == True:
        print(separator)
        print("Test sample dataframe for all:")
        print(test_sample["all"])
        print(test_sample["all"].shape)
        print(separator)
        print("Names sample dataframe for all:")
        print(names_sample["all"])
        print(names_sample["all"].shape)
        print(separator)

    # SHAP will run forever if you give the entire the training dataset. We use k-means to reduce the training dataset into specific centroids
    background = shap.kmeans(X_train, K_MEANS_CLUSTERS)

    if DEBUG == True:
        print("Number of k-means clusters:")
        print(K_MEANS_CLUSTERS)
        print(separator)

    # Algorithms to consider for interpretations
    algorithm = "randomforest_binary"

    # A dictionary to hold trained models
    model_gs = {}
    model_explainer = {}

    if DEBUG == True:
        print("Execution for algorithm: ", algorithm)

    # Train the machine/deep learning model
    model_temp = train_model(X_train, y_train)
    model_gs[algorithm] = model_temp
    # Evaluate the machine/deep learning model
    evaluate_model(model_gs[algorithm], X_test, y_test)

    for family in per_category_test.keys():
        # Get accuracy calculations on testing dataset per malware family
        evaluate_model_on_family(model_gs[algorithm], family, test_sample[family], algorithm)

    # We will derive explanations using the Kernel Explainer
    model_explainer[algorithm] = shap.KernelExplainer(model_gs[algorithm].predict, background)

    selected_features = df[["Reputation", "Length", "Words_Mean", "Max_Let_Seq", "Words_Freq", "Vowel_Freq",
                            "Entropy", "DeciDig_Freq", "Max_DeciDig_Seq"]]

    selected_families = ["all", "all_DGAs", "bamital", "matsnu"]

    for family in selected_families:
        print("Calculating SHAP values for family:", family)

        model_shap_values = model_explainer[algorithm].shap_values(test_sample[family])
        model_shap_values = np.asarray(model_shap_values)

        explain_with_shap_summary_plots(model_gs[algorithm], model_shap_values, family, test_sample[family], algorithm)
        explain_with_shap_dependence_plots(model_gs[algorithm], model_shap_values, family, test_sample[family],
                                           selected_features, algorithm)
        explain_with_force_plots(model_gs[algorithm], model_shap_values, family, test_sample[family],
                                 names_sample[family], algorithm, model_explainer[algorithm])