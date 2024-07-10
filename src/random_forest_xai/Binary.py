# True to print debugging outputs, False to silence the program
DEBUG = True
DRIFT = False
separator = "-------------------------------------------------------------------------"
# Correlation threshold for Pearson correlation. For feature pairs with correlation higher than the threshold, one feature is dropped
CORRELATION_THRESHOLD = 0.9
# Define the number of clusters that will represent the training dataset for SHAP framework (cannot give all training samples)
K_MEANS_CLUSTERS = 100

# Import the necessary libraries (tested for Python 3.11)
import numpy as np
import pickle
import os
import sys
import shap
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Families considered for SHAP interpretations
families = ["tranco", "bamital", "banjori", "matsnu", "dyre", "conficker", "cryptolocker", "suppobox", "nymaim", "pykspa",
            "bedep", "beebone", "blackhole", "bobax", "ccleaner",
            "chinad", "chir",  "corebot",  "darkshell", "diamondfox", "dircrypt",
            "dmsniff", "dnsbenchmark", "dnschanger", "downloader", "ebury", "ekforward", "emotet",
            "feodo", "fobber", "gameover", "gozi", "goznym", "gspy", "hesperbot", "infy",
            "locky", "madmax", "makloader",  "mirai", "modpack", "monerominer", "murofet",
            "murofetweekly", "mydoom", "necurs", "nymaim2",  "oderoor", "omexo", "padcrypt",
            "pandabanker", "pitou", "proslikefan", "pushdo", "pushdotid", "pykspa2", "pykspa2s",
            "qadars", "qakbot", "qhost", "qsnatch", "ramdo", "ramnit", "ranbyus", "randomloader", "redyms", "rovnix",
            "shifu", "simda", "sisron", "sphinx",  "sutra", "symmi", "szribi", "tempedreve",
            "tempedrevetdd", "tinba", "tinynuke", "tofsee", "torpig", "tsifiri", "ud2", "ud3", "ud4", "urlzone",
            "vawtrak", "vidro", "vidrotid", "virut", "volatilecedar", "wd", "xshellghost", "xxhex"]


RNG = np.random.RandomState(seed=42)


def load_dataset(filename, families):
    # Load the dataset in the form of a csv
    df = pd.read_csv(filename)
    headers = pd.read_csv(filename, index_col=False, nrows=0).columns.tolist()
    features = headers[0:-3]

    family_counts = df['Family'].value_counts()
    if DEBUG:
        print("The number of samples for each family:", family_counts)

    # Filter classes with counts greater than or equal to 100
    valid_classes = family_counts[family_counts >= 3000].index
    families = [item for item in families if item in valid_classes]

    # Filter the DataFrame to keep only rows with valid classes
    df = df[df['Family'].isin(valid_classes)]

    family_counts = df['Family'].value_counts()
    if DEBUG:
        print("NOW HE HAVE", len(families), families)
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


def split_dataset(df, df_test):
    if df_test.equals(df):
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=RNG, shuffle=True)
    else:
        train_set, _ = train_test_split(df, test_size=0.2, random_state=RNG, shuffle=True)
        _, test_set = train_test_split(df_test, test_size=0.2, random_state=RNG, shuffle=True)
        test_set = test_set.sample(n=30000)

    # Split features from labels (the last three columns are domain name, binary label, malware family)
    X_train = train_set.iloc[:, :-3]
    y_train = train_set.iloc[:, -2:-1]
    X_test = test_set.iloc[:, :-3]
    y_test = test_set.iloc[:, -3:]

    return X_train, y_train, X_test, y_test


def scale_dataset(X_train, X_test):
    # Calculate minimum and maximum values using only the training dataset
    minimum = X_train.min()
    maximum = X_train.max()

    # Scale the training dataset
    X_train_scaled = (X_train - minimum) / (maximum - minimum)

    # Scale the testing dataset using the same minimum and maximum values
    X_test_scaled = (X_test - minimum) / (maximum - minimum)

    # Return the scaled training and testing datasets
    return X_train_scaled, X_test_scaled, minimum, maximum


def oversample_data(X_train, y_train):
    # Oversample the data using SMOTE
    sm = SMOTE(random_state=RNG)
    print("COUNT: count for each family before oversampling", y_train.value_counts())
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print("COUNT: count for each family after oversampling", y_train.value_counts())

    return X_train, y_train


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=50, max_depth=20, n_jobs=-1, random_state=RNG)
    model.fit(X_train, y_train.values.ravel())

    return model


def evaluate_model(model, X_test, y_test, algorithm):
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test["Label"].values, predictions, normalize=True)
    prec = precision_score(y_test["Label"].values, predictions)
    rec = recall_score(y_test["Label"].values, predictions)
    f1 = f1_score(y_test["Label"].values, predictions)

    # Print the different testing scores
    print("Algorithm: ", algorithm)
    print("Accuracy: ", acc)
    print("Precision: ", prec)
    print("Recall: ", rec)
    print("F1 score: ", f1)
    print(separator)

    return acc, prec, rec, f1


def explain_with_shap_summary_plots(model_shap_values, family, X, version):
    # Plot bar summary plot using SHAP values
    prepend_path = os.path.join("..", "..", "files", "results", "TreeClassifier", "binary_final", version, str(family),
                                "summary-plots")
    command = "mkdir " + prepend_path
    subprocess.run(command, shell=True)

    if family == "all":
        fig = plt.clf()
        shap.summary_plot(model_shap_values, X, plot_type="bar", show=False, feature_names=X.columns)
        name = os.path.join(prepend_path, f"{family}-summarybar.png")
        plt.savefig(name)
        plt.close("all")

    else:
        # Plot summary plot using SHAP values
        fig = plt.clf()
        shap.summary_plot(model_shap_values[1], X, plot_type="dot", show=False, feature_names=X.columns)
        name = os.path.join(prepend_path, f"{family}-beeswarm.png")
        plt.savefig(name)
        plt.close("all")

    return None

def explain_with_force_plots(model, model_shap_values, family, X, X_names, model_explainer,
                             version):
    # Plot force plots using SHAP values (local explanations)
    prepend_path = os.path.join("..", "..", "files", "results", "TreeClassifier", "binary_final", version, str(family),
                                "force-plots")
    command = "mkdir " + prepend_path
    subprocess.run(command, shell=True)

    predictions = model.predict(X)
    index_values = list(X.index.values)
    sequence = 0
    for index in index_values:
        original_name = X_names[index]
        name = original_name.replace(".", "+")
        prediction = predictions[sequence]

        fig = plt.clf()
        shap.force_plot(model_explainer.expected_value[0], model_shap_values[0][sequence, :], X.loc[index],
                        matplotlib=True, show=False)
        name_of_file = os.path.join(prepend_path,
                                    f"{family}-force-name-{name}-prediction-{prediction}.png")
        plt.title(original_name, y=1.5)
        plt.savefig(name_of_file, bbox_inches='tight')
        plt.close("all")

        sequence += 1
        # Plot only the first 100 or less if no more than 100 exist
        if sequence == 100:
            break

    return None


if __name__ == "__main__":
    plot_scores = [["Year", "Accuracy", "Precision", "Recall", "F1-Score"]]
    if DRIFT:
        loop = range(0, 10)
    else:
        loop = range(0, 1)

    for i in loop:
        # Dataset to load
        if DRIFT:
            filename = "../../files/labeled_datasets_features/binary/binary_features_2010_random.csv"
            filename_test = f"../../files/labeled_datasets_features/binary/binary_features_201{i}_random.csv"
            print(f"2010 vs 201{i}")
            # Load the dataset
            df, features, families = load_dataset(filename, families)
            df_test, features_test, families = load_dataset(filename_test, families)

        else:
            filename = "../../files/labeled_datasets_features/binary/binary_features.csv"
            df, features, families = load_dataset(filename, families)
            families_mapping = {family: index for index, family in enumerate(families)}
            print(families_mapping)

        if DEBUG:
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

        if DRIFT:
            df_test = df_test.drop(columns=to_drop, inplace=False)

        if DEBUG:
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
        if DRIFT:
            X_train, y_train, X_test, y_test = split_dataset(df, df_test)
        else:
            X_train, y_train, X_test, y_test = split_dataset(df, df)

        if DEBUG:
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
        X_train, X_test, minimum, maximum = scale_dataset(X_train, X_test)

        if DEBUG:
            print("Scaled X_train:")
            print(X_train)
            print(separator)
            print("Scaled X_test:")
            print(X_test)
            print(separator)

        # Data oversampling to deal with class imbalance
        X_train, y_train = oversample_data(X_train, y_train)

        if DEBUG:
            print("Size of oversampled X_train:")
            print(len(X_train))
            print(separator)
            print("Size of oversampled y_train:")
            print(len(y_train))
            print(separator)



        if not DRIFT:
            versions = ["v1", "v2", "v3", "v4"]

            ALL = {}
            X = {}
            y = {}
            X_names = {}
            test = {}
            test_names = {}
            temp = {}
            temp_names = {}
            stat = {}

            for v in versions:
                with open(f"test_sample_200/X_{v}", "rb") as file:
                    X[v] = pickle.load(file)

                with open(f"test_sample_200/X_names_{v}", "rb") as file:
                    X_names[v] = pickle.load(file)

                with open(f"test_sample_200/all_{v}", "rb") as file:
                    ALL[v] = pickle.load(file)

                temp["tranco"] = ALL[v][ALL[v]["Family"] == 0]
                for fam in range(1, 10):
                    for key, value in families_mapping.items():
                        if value == fam:
                            family = key
                    temp[family] = ALL[v][ALL[v]["Family"] == fam]
                    temp[family] = pd.concat([temp[family], temp["tranco"]], axis=0)
                    temp_names[family] = temp[family]["Name"]
                    temp[family] = temp[family].iloc[:, :-3]

                temp_names["tranco"] = temp["tranco"]["Name"]
                temp["tranco"] = temp["tranco"].iloc[:, :-3]

                test[v] = temp
                test_names[v] = temp_names

                test_df = pd.DataFrame(X[v])
                is_included = test_df.isin(X_train).all().all()
                print(separator)
                print("CKECK!!! (if the pickle is included in the training set)", is_included)
                print(separator)


            # SHAP will run forever if you give the entire the training dataset. We use k-means to reduce the training dataset into specific centroids
            background = shap.kmeans(X_train, K_MEANS_CLUSTERS)
            background = np.array(background.data)

            if DEBUG:
                print("Number of k-means clusters:")
                print(K_MEANS_CLUSTERS)
                print(separator)

        # Algorithms to consider for interpretations
        algorithm = "randomforest_binary"

        if DEBUG:
            print("Execution for algorithm: ", algorithm)

        # Train the machine/deep learning model
        model = train_model(X_train, y_train)
        print(sys.getsizeof(model))
        # Evaluate the machine/deep learning model
        acc, prec, rec, f1 = evaluate_model(model, X_test, y_test, algorithm)
        plot_scores.append([f"201{i}", acc, prec, rec, f1])

        if not DRIFT:
            # We will derive explanations using the Kernel Explainer
            model_explainer = shap.TreeExplainer(model=model, data=background)

            for v in versions:
                # print("Calculating SHAP values for family:", family, " version: ", v)
                print("This is the test sample:\n", X[v])

                model_shap_values = model_explainer.shap_values(X[v].sample(n=200))
                explain_with_shap_summary_plots(model_shap_values, "all", X[v].sample(n=200),  v)

                for fam in range(0, 1):
                    for key, value in families_mapping.items():
                        if value == fam:
                            family = key

                    print("Calculating SHAP values for family: ", family)
                    print(separator)
                    print(test[v][family])
                    model_shap_values = model_explainer.shap_values(test[v][family])
                    print(model_shap_values)
                    print(len(model_shap_values))
                    explain_with_shap_summary_plots(model_shap_values, family, test[v][family], v)
                    explain_with_force_plots(model, model_shap_values, family, test[v][family], test_names[v][family],
                                                model_explainer, v)

    if DRIFT:
        # Convert list to DataFrame
        df = pd.DataFrame(data=plot_scores[1:], columns=plot_scores[0])
        df = df.round(3)

        color_map = {
            "Accuracy": "blue",
            "Precision": "green",
            "Recall": "red",
            "F1-Score": "yellow"
        }

        # Plot using Matplotlib
        plt.figure(figsize=(10, 6))  # Adjust figure size as needed
        for metric in ["Accuracy", "Precision", "Recall", "F1-Score"]:
            plt.plot(df["Year"], df[metric], label=metric, color=color_map[metric], marker='o')

        # plt.title('Model Evaluation Metrics Over Years')
        plt.xlabel('Year')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)

        # Create directory if it doesn't exist
        prepend_path = os.path.join("..", "..", "files", "results", "binary", "concept_drift_new")
        command = "mkdir " + prepend_path
        subprocess.run(command, shell=True)

        # Save the plot
        name = os.path.join(prepend_path, "train_2010_drift_random.png")
        plt.savefig(name)

        # Close the plot
        plt.close()