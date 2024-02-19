import sys
import os

DEBUG = "True"
separator = "----------------------------------------"

# What will be included in the final file: Tranco names and DGA's from various families
dgas = ["tranco", "bamital", "banjori", "bedep", "beebone", "blackhole", "bobax", "ccleaner",
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

if DEBUG == "True":
    print("These name categories will be included in the final dataset")
    print(dgas)
    print(separator)

# Assign a number to each entry of the dgas list so that they can be discerned
dga_mapping = {family: index for index, family in enumerate(dgas)}

if DEBUG == "True":
    print("The name categories were assigned to the following numbers")
    print(dga_mapping)
    print(separator)

# The filename where everything will be stored
filename_out = "../../files/labeled_datasets_features/labeled_dataset_multiclass_20Κ.csv"
fdw = open(filename_out, "w")

total_sizes = []

# How many names to keep from each DGA family (not Tranco, but DGA)
maximum_size = 20000

if DEBUG == "True":
    print("Each DGA family will be included with the following number of names: ")
    print(maximum_size)
    print(separator)

if DEBUG == "True":
    print("Loading the Mozilla Firefox suffixes in a set")
    print(separator)

# Load Mozilla Firefox libraries to exclude valid domain name suffixes
suffix_list = "../../files/suffixes.txt"
# Load suffixes in a set
# This should be outside the loop (TO-DO)
fdr = open(suffix_list, "r", encoding="utf-8")
suffixes = set()
for line in fdr:
    suffix = line.strip()
    suffixes.add(suffix)

fdr.close()


for dga in dgas:
    if DEBUG == "True":
        print("Now working on: ", dga)

    if dga == "tranco":
        # Load tranco names
        filename = "../../files/tranco_filtered_files/tranco_remaining.txt"
    else:
        # Load names of a specific DGA family
        filename = "../../files/dga_20K/" + str(dga) + "_dga.csv"

    fdr = open(filename, "r", encoding="utf-8")

    # Find the appropriate prefix
    prefix_set = set()
    for line in fdr:
        line = line.strip()
        name = line.split(",")[0].replace('"', '')
        labels = name.split(".")
        labels.reverse()
        candidate_suffix = labels[0]
        index = 0
        try:
            while candidate_suffix in suffixes:
                index += 1
                candidate_suffix = labels[index] + "." + candidate_suffix
            labels.reverse()
            prefix = ".".join(labels[0:(len(labels) - index)])
            to_add = str(prefix) + "," + str(name)
            prefix_set.add(to_add)
        except:
            pass
        if (dga == "tranco" and len(prefix_set) == maximum_size) or (dga != "tranco" and len(prefix_set) == maximum_size):
            break

    fdr.close()

    # Labeling for binary classifiers: 0 for tranco (legitimate) and 1 for DGA's
    for item in prefix_set:
        if dga == "tranco":
            item = item + ",0," + str(dga)
        else:
            item = item + ",1," + str(dga)
        fdw.write(item + "\n")

    total_names = len(prefix_set)
    total_sizes.append(total_names)

if DEBUG == "True":
    print("The total size of each name category is:")
    print(total_sizes)

compound_list = []
start_value = 0
for item in total_sizes:
    start_value += int(item)
    compound_list.append(start_value)

fdw.close()
