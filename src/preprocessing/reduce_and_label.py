import sys
import os
import random

DEBUG = True
RANDOM = False
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

dgas_2010 = ['tranco', 'gozi', 'conficker', 'murofet', 'vidro', 'gameover', 'bamital', 'szribi', 'torpig', 'mydoom', 'murofetweekly']

dgas_2015 = ['tranco', 'nymaim', 'sisron', 'torpig', 'madmax', 'necurs', 'oderoor', 'sutra', 'pushdo', 'conficker', 'chinad',
             'dyre', 'blackhole', 'ranbyus', 'tempedrevetdd', 'tofsee', 'pitou', 'corebot', 'mydoom', 'symmi', 'pykspa',
             'virut', 'cryptolocker', 'qakbot', 'ud2', 'qadars', 'ekforward', 'murofetweekly', 'suppobox', 'vidro',
             'bamital', 'emotet', 'infy', 'locky', 'modpack', 'murofet', 'matsnu', 'bedep', 'gameover', 'xshellghost',
             'szribi', 'gozi', 'proslikefan']

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
# filename_out = "../../files/labeled_dataset/labeled_dataset_binary.csv"
# fdw = open(filename_out, "w")

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

years = ["2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019"]
for year in years:
    filename_out = "../../files/labeled_dataset/multiclass/labeled_dataset_multiclass_20K.csv"
    fdw = open(filename_out, "w")
    for dga in dgas:
        if DEBUG:
            print("Now working on: ", dga)

        if dga == "tranco":
            # Load tranco names
            filename = "../../files/tranco_filtered_files/tranco_remaining.txt"
        else:
            # Load names of a specific DGA family
            filename = "../../files/dga_full_by_year/" + str(dga) + "_dga_" + year + ".csv"

        # fdr = open(filename, "r", encoding="utf-8")

        try:
            with open(filename, 'r', encoding='utf-8') as fdr:
                # Find the appropriate prefix
                prefix_set = set()

                if RANDOM:
                    total_lines = sum(1 for _ in fdr)
                    # Seek back to the beginning of the file
                    fdr.seek(0)

                    # Calculate how many lines you want to sample
                    max_target = 0000 if dga == 'tranco' else 20000
                    p = max_target / total_lines

                for line in fdr:
                    if RANDOM:
                        variable = True if random.random() <= p else False
                    else:
                        variable = True

                    if variable:
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
                        if not RANDOM:
                            if (dga == "tranco" and len(prefix_set) == maximum_size) or (
                                    dga != "tranco" and len(prefix_set) == maximum_size):
                                break
                    else:
                        pass

                # Labeling for binary classifiers: 0 for tranco (legitimate) and 1 for DGA's
                for item in prefix_set:
                    if dga == "tranco":
                        item = item + ",0," + str(dga)
                    else:
                        item = item + ",1," + str(dga)
                    fdw.write(item + "\n")

        except FileNotFoundError:
            print(f"File not found: {filename}. Skipping...")
            # You can add additional handling here if needed
            pass


        fdr.close()

fdw.close()
