# Help from: https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python
# Code from Explainable AI with Python

# True to print debugging outputs, False to silence the program
DEBUG = True
STAT = False
DRIFT = False
separator = "-------------------------------------------------------------------------"
# Correlation threshold for Pearson correlation. For feature pairs with correlation higher than the threshold, one feature is dropped
CORRELATION_THRESHOLD = 0.9
# Define the number of clusters that will represent the training dataset for SHAP framework (cannot give all training samples)
K_MEANS_CLUSTERS = 100
# Define the number of testing samples on which SHAP will derive interpretations
SAMPLES_NUMBER = 500

# Import the necessary libraries (tested for Python 3.9)
import numpy as np
import pickle
import os
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
    X = df.drop(['Family', 'Label', 'Name'], axis=1)

    family_counts = df['Family'].value_counts()
    print("The number of samples for each family: \n", family_counts)

    # Filter classes with counts greater than or equal to 100
    valid_classes_100 = family_counts[family_counts >= 100].index
    valid_classes_1000 = family_counts[family_counts >= 1000].index
    valid_classes_2000 = family_counts[family_counts >= 2000].index
    valid_classes_3000 = family_counts[family_counts >= 3000].index
    families = [item for item in families if item in valid_classes_3000]
    print("NOW HE HAVE", len(families), "families")

    # Filter the DataFrame to keep only rows with valid classes
    df = df[df['Family'].isin(valid_classes_3000)]

    family_counts = df['Family'].value_counts()
    print("The number of samples for each family:", family_counts)

    # Return a dataframe and the names of the features
    return df, features, families, X


def drop_features_by_correlation(df):
    # Calculate correlation coefficients for pairs of features
    df_for_corr = df.drop(labels=['Name', 'Label', 'Family'], axis=1)
    correlation_coeffs = df_for_corr.corr()

    # Keep the upper triangular matrix of correlation coefficients
    upper_tri = correlation_coeffs.where(np.triu(np.ones(correlation_coeffs.shape), k=1).astype(np.bool_))

    # Drop columns with high correlation (one of the features consisting the pair is dropped, the other is kept)
    to_drop = [column for column in upper_tri.columns if any(abs(upper_tri[column]) >= CORRELATION_THRESHOLD)]

    if DEBUG:
        print("Correlation threshold is:")
        print(CORRELATION_THRESHOLD)
        print(separator)

    df = df.drop(columns=to_drop, inplace=False)
    features = df.columns.tolist()[0:-3]

    # Return the names of the dropped features, the new dataframe and the names of the new features within the feature set
    return to_drop, df, features


def split_dataset(df, df_test, families):
    # Split the dataset into training and testing sets
    if df_test.equals(df):
        train_set, test_set = train_test_split(df, test_size=0.2, random_state=RNG, shuffle=True)
    else:
        train_set, _ = train_test_split(df, test_size=0.2, random_state=RNG, shuffle=True)
        _, test_set = train_test_split(df_test, test_size=0.2, random_state=RNG, shuffle=True)

    # Split features from labels (the last three columns are domain name, binary label, malware family)
    X_train = train_set.iloc[:, :-3]
    y_train = train_set.iloc[:, -1:]
    X_test = test_set.iloc[:, :-3]
    y_test = test_set.iloc[:, -3:]

    print("Y TRAIN: before mapping", y_train)
    families_mapping = {family: index for index, family in enumerate(families)}
    print("Y TEST: before mapping", y_test.iloc[:, -1:])
    print("The mapping:\n", families_mapping)

    y_train_original = y_train
    y_train = y_train_original.replace(families_mapping)
    y_test.iloc[:, -1:] = y_test.iloc[:, -1:].replace(families_mapping)

    print("Y TRAIN: after mapping", y_train)
    print("Y TEST: after mapping", y_test.iloc[:, -1:])

    return X_train, y_train, X_test, y_test, families_mapping


def scale_dataset(X_train, X_test):
    # Scale the dataset using min-max scaling
    minimum = X_train.min()
    maximum = X_train.max()
    X_train = (X_train - minimum) / (maximum - minimum)
    X_test = (X_test - minimum) / (maximum - minimum)

    # Return the scaled training and testing datasets
    return X_train, X_test, minimum, maximum

def reverse_scale_dataset(X_scaled, minimum, maximum):
    # Reverse the scaling process
    X_reversed = X_scaled * (maximum - minimum) + minimum
    return X_reversed


def oversample_data(X_train, y_train):
    # Oversample the data using SMOTE
    sm = SMOTE(random_state=RNG)

    print("COUNT: count for each family before oversampling", y_train.value_counts())
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print("COUNT: count for each family after oversampling", y_train.value_counts())

    return X_train, y_train


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=10, max_depth=10, n_jobs=-1, random_state=RNG)
    model.fit(X_train, y_train.values.ravel())

    return model


def evaluate_model(model, X_test, y_test, algorithm):
    predictions = model.predict(X_test)

    # Convert both arrays to int
    arr1 = predictions.astype(int)
    arr2 = y_test["Family"].values.astype(int)

    acc = accuracy_score(arr2, arr1, normalize=True)
    prec = precision_score(arr2, arr1, average='macro', zero_division=1)
    rec = recall_score(arr2, arr1, average='macro')
    f1 = f1_score(arr2, arr1, average='macro')

    # Print the different testing scores
    print("Algorithm: ", algorithm)
    print("Accuracy: ", acc)
    print("Precision: ", prec)
    print("Recall: ", rec)
    print("F1 score: ", f1)
    print(separator)

    return acc, prec, rec, f1


def split_testing_dataset_into_categories(X_test, y_test, families_mapping):
    # For SHAP purposes, to derive interpretations per malware family, we will split the testing dataset per malware family
    test_merged = pd.merge(left=X_test, left_index=True, right=y_test, right_index=True, how='inner')

    per_category_test = {}

    # For all DGA's regardless of the malware family
    all_test = test_merged.iloc[:, :-2]
    per_category_test["all"] = all_test

    per_category_test["all_DGAs"] = test_merged[test_merged.iloc[:, -1] != families_mapping["tranco"]].iloc[:, :-2]

    for family in families:
        test = test_merged[test_merged.iloc[:, -1] == families_mapping[str(family)]]
        test = test.iloc[:, :-2]
        if len(test) > 100:
            per_category_test[str(family)] = test

    per_category_test["hash-based"] = pd.concat([per_category_test["bamital"], per_category_test["dyre"]],
                                                ignore_index=True)
    per_category_test["wordlist-based"] = pd.concat(
        [per_category_test["matsnu"], per_category_test["suppobox"], per_category_test["pykspa"]], ignore_index=True)
    per_category_test["arithmetic-based"] = pd.concat(
        [per_category_test["banjori"], per_category_test["cryptolocker"], per_category_test["conficker"],
         per_category_test["nymaim"], per_category_test["pykspa"]], ignore_index=True)

    print("HASH BASED", per_category_test["hash-based"], "\n")
    print("WORDLIST BASED", per_category_test["wordlist-based"], "\n")
    print("ARITHMETIC BASED", per_category_test["arithmetic-based"], "\n")

    # This will hold testing samples per malware family, e.g. per_category_test["bamital"] holds testing samples for bamital
    return per_category_test


def explain_with_shap_summary_plots(model_shap_values, X, version, families_mapping):
    # Plot bar summary plot using SHAP values
    prepend_path = os.path.join("..", "..", "files", "results", "TreeClassifier", "multiclass_new", version,
                                "all", "summary-plots")
    command = "mkdir" + prepend_path
    subprocess.run(command, shell=True)

    fig = plt.clf()
    shap.summary_plot(model_shap_values[:10], X.values, plot_type="bar", class_names={0: "tranco", 1: "bamital",
                                                                               2: "banjori", 3: "matsnu", 4: "dyre",
                                                                               5: "conficker", 6: "cryptolocker",
                                                                               7: "suppobox",
                                                                               8: "nymaim", 9: "pykspa"},
                      feature_names=features_red, show=False)
    name = os.path.join(prepend_path, f"first10-summarybar.png")
    plt.savefig(name)
    plt.close("all")

    fig = plt.clf()
    shap.summary_plot(model_shap_values, X.values, plot_type="bar", feature_names=features_red, show=False)
    name = os.path.join(prepend_path, f"all-summarybar.png")
    plt.savefig(name)
    plt.close("all")


    for fam in range (0,10):
        for key, value in families_mapping.items():
            if value == fam:
                family = key
        # Plot bar summary plot using SHAP values
        prepend_path = os.path.join("..", "..", "files", "results", "TreeClassifier", "multiclass_new",
                                    version, str(family), "summary-plots")
        command = "mkdir" + prepend_path
        subprocess.run(command, shell=True)

        # Plot summary plot using SHAP values
        fig = plt.clf()
        shap.summary_plot(model_shap_values[fam], X, show=False, feature_names=X.columns)
        name = os.path.join(prepend_path, f"{family}-beeswarm.png")
        plt.savefig(name)
        plt.close("all")


def explain_with_shap_dependence_plots(model_shap_values, X, features, version, families_mapping):
    for fam in range(0, 10):
        for key, value in families_mapping.items():
            if value == fam:
                family = key
        # Plot dependence plot using SHAP values for multiple features
        prepend_path = os.path.join("..", "..", "files", "results", "TreeClassifier", "multiclass_new",
                                    version, str(family) , "dependence-plots")
        command = "mkdir " + prepend_path
        subprocess.run(command, shell=True)

        for feature in features:
            fig = plt.clf()
            shap.dependence_plot(ind=feature, shap_values=model_shap_values[fam], features=X, show=False)
            name = os.path.join(prepend_path, f"{family}-dependence-{feature}.png")
            plt.savefig(name, bbox_inches='tight')
            plt.close("all")

    return None


def explain_with_force_plots(model, model_shap_values, ALL, X_names, model_explainer, version, families_mapping):
    for fam in range(0, 10):
        for key, value in families_mapping.items():
            if value == fam:
                family = key
        # Plot force plots using SHAP values (local explanations)
        prepend_path = os.path.join("..", "..", "files", "results", "TreeClassifier", "multiclass_new",
                                    version, str(family) , "force-plots")
        command = "mkdir " + prepend_path
        subprocess.run(command, shell=True)

        X = ALL[ALL["Family"] == fam]
        X = X.iloc[:, :-3]
        predictions = model.predict(X)
        # X = X[X["Family"]==fam]
        # X = X.iloc[:, :-3]
        index_values = list(X.index.values)
        sequence = 0
        for index in index_values:
            original_name = X_names[index]
            name = original_name.replace(".", "+")
            prediction = predictions[sequence]

            fig = plt.clf()
            shap.force_plot(model_explainer.expected_value[fam], model_shap_values[fam][sequence, :], X.loc[index],
                            matplotlib=True, show=False)
            name_of_file = os.path.join(prepend_path,
                                        f"{family}-force-name-{name}-prediction-{prediction}.png")
            plt.title(original_name, y=1.5)
            plt.savefig(name_of_file, bbox_inches='tight')
            plt.close("all")

            sequence += 1
            # Plot only the first 100 or less if no more than 100 exist
            if sequence == 50:
                break

    return None


if __name__ == "__main__":
    plot_scores = [["Year", "Accuracy", "Precision", "Recall", "F1-Score"]]
    if DRIFT:
        loop = range(5, 10)
    else:
        loop = range(0, 1)

    for i in loop:
        # Dataset to load
        if DRIFT:
            filename = "../../files/labeled_datasets_features/multiclass/multiclass_features_2010_onlyclassesfrom2010.csv"
            filename_test = f"../../files/labeled_datasets_features/multiclass/multiclass_features_201{i}.csv"
            print(f"2015 vs 201{i}")
            # Load the dataset
            df, features, families, _ = load_dataset(filename, families)
            df_test, features_test, families, _ = load_dataset(filename_test, families)

        else:
            filename = "../../files/labeled_datasets_features/multiclass/multiclass_features_20K.csv"
            df, features, families, X = load_dataset(filename, families)

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
        to_drop, df, features_red = drop_features_by_correlation(df)
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
            print(features_red)
            print(separator)

        # Split dataset into training and testing portions
        if DRIFT:
            X_train, y_train, X_test, y_test, families_mapping = split_dataset(df, df_test, families)
        else:
            X_train, y_train, X_test, y_test, families_mapping = split_dataset(df, df, families)

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
            # all_v1 = []
            # all_v2 = []
            # all_v3 = []
            # all_v4 = []
            #
            ALL = {}
            X = {}
            y = {}
            X_names = {}

            # all = pd.concat([X_test, y_test], axis=1)
            #
            # for f in all["Family"].unique():
            #     temp = all[all["Family"]==f].sample(n=300, random_state=1111)
            #     all_v1.append(temp)
            #
            #     temp = all[all["Family"]==f].sample(n=300, random_state=2222)
            #     all_v2.append(temp)
            #
            #     temp = all[all["Family"]==f].sample(n=300, random_state=3333)
            #     all_v3.append(temp)
            #
            #     temp = all[all["Family"]==f].sample(n=300, random_state=4444)
            #     all_v4.append(temp)
            #
            # ALL_v1 = pd.concat(all_v1)
            # ALL["v1"] = ALL_v1
            # ALL_v2 = pd.concat(all_v2)
            # ALL["v2"] = ALL_v2
            # ALL_v3 = pd.concat(all_v3)
            # ALL["v3"] = ALL_v3
            # ALL_v4 = pd.concat(all_v4)
            # ALL["v4"] = ALL_v4

            for v in versions:
                # X[v] = ALL[v].iloc[:, :-3]
                # X_names[v] = ALL[v]["Name"]
                # y[v] = ALL[v].iloc[:, -3:]
                #
                # with open(f"test_sample_300/all_{v}", "wb") as file:
                #     pickle.dump(ALL[v], file)
                #
                # with open(f"test_sample_300/X_{v}", "wb") as file:
                #     pickle.dump(X[v], file)
                #
                # with open(f"test_sample_300/X_names_{v}", "wb") as file:
                #     pickle.dump(X_names[v], file)

                with open(f"test_sample_300/X_{v}", "rb") as file:
                    X[v] = pickle.load(file)

                with open(f"test_sample_300/X_names_{v}", "rb") as file:
                    X_names[v] = pickle.load(file)

                with open(f"test_sample_300/all_{v}", "rb") as file:
                    ALL[v] = pickle.load(file)


                test_df = pd.DataFrame(X[v])
                is_included = test_df.isin(X_train).all().all()
                print(separator)
                print(ALL[v]['Family'].value_counts())
                print("CKECK!!! (if the pickle is included in the training set)", is_included)
                print(separator)

                if STAT:
                    for fam in range(0, 10):
                        for key, value in families_mapping.items():
                            if value == fam:
                                family = key
                        temp_stat = ALL[v][ALL[v]["Family"] == fam]
                        stat = reverse_scale_dataset(temp_stat, minimum, maximum)
                        # Create directory if it doesn't exist
                        prepend_path = os.path.join("..", "..", "files", "results", "TreeClassifier", "multiclass",
                                                        "stats", v, family)
                        command = "mkdir " + prepend_path
                        subprocess.run(command, shell=True)

                        for feature in features_red:
                            plt.figure(figsize=(8, 6))
                            plt.hist(stat[feature], bins=25, color='green', edgecolor='black')
                            plt.title('Histogram of ' + feature + ' for ' + family + ': ' + v)
                            plt.xlabel(feature)
                            plt.ylabel('Domain Names')
                            plt.grid(True)

                            # Save the plot
                            name = os.path.join(prepend_path, f"{family}_{feature}_{v}")
                            plt.savefig(name)

                            # Close the plot
                            plt.close()

            # SHAP will run forever if you give the entire the training dataset. We use k-means to reduce the training dataset into specific centroids
            # background = shap.kmeans(X_train, K_MEANS_CLUSTERS)
            # print(type(background))
            # background = np.array(background.data)

            if DEBUG:
                print("Number of k-means clusters:")
                print(K_MEANS_CLUSTERS)
                print(separator)

        # Algorithms to consider for interpretations
        algorithm = "randomforest_multiclass"

        if DEBUG:
            print("Execution for algorithm: ", algorithm)

        # Train the machine/deep learning model
        model = train_model(X_train, y_train)
        # Evaluate the machine/deep learning model
        acc, prec, rec, f1 = evaluate_model(model, X_test, y_test.iloc[:, -1:], algorithm)
        plot_scores.append([f"201{i}", acc, prec, rec, f1])

        if not DRIFT:
            # We will derive explanations using the Kernel Explainer
            model_explainer = shap.TreeExplainer(model)

            selected_features = df[["Reputation", "Length", "Words_Mean", "Max_Let_Seq", "Words_Freq",
                                    "Vowel_Freq", "Entropy", "Max_DeciDig_Seq", "Freq_0"]]


            for v in versions:
                print("Calculating SHAP values for version: ", v)

                print(X[v])
                model_shap_values = model_explainer.shap_values(X[v])

                explain_with_shap_summary_plots(model_shap_values, X[v], v, families_mapping)
                # explain_with_shap_dependence_plots(model_shap_values, X[v], features_red, v, families_mapping)
                # explain_with_force_plots(model, model_shap_values, ALL[v], X_names[v],
                #                             model_explainer, v, families_mapping)



    if DRIFT:
        # Convert list to DataFrame
        df = pd.DataFrame(data=plot_scores[1:], columns=plot_scores[0])

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
        prepend_path = os.path.join("..", "..", "files", "results", "multiclass", "concept_drift")
        command = "mkdir " + prepend_path
        subprocess.run(command, shell=True)

        # Save the plot
        name = os.path.join(prepend_path, "train_2015_drift.png")
        plt.savefig(name)

        # Close the plot
        plt.close()