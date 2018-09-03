import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import defaultdict

def shuffle_split_balance_samples(meta, Y, X, samples_per_class):
    dicpos, dicsam, dicspl = [defaultdict(list) for i in range(3)]
    ltr, lte = [[] for i in range(2)]
    classes, counts = np.unique(Y, return_counts=True)

    # avg = int(np.divide(nsamples, len(classes)))

    # Find samples belonging to each class
    print("Finding samples per class")
    for classs in classes:
        dicpos[classs] = np.where(Y==classs)[0]

    # For each class, add to a list its associated samples
    print("Linking samples to a class")
    for key in sorted(dicpos.keys()):
        for idx in dicpos[key]:
            newrow = meta[idx, :].tolist() + [Y[idx]] + X[idx,:].tolist()
            dicsam[key].append(newrow)

    # Shuffle and split the samples per class
    print("Shuffling and splitting data in train/test")
    for key in sorted(dicsam.keys()):
        np.random.shuffle(dicsam[key])
        ltr_i, lte_i = split_samples(dicsam[key], samples_per_class)
        tup = (ltr_i, lte_i)
        dicspl[key] = tup

    # Balance the number of samples per class
    print("Balancing number of samples per class")
    for key in sorted(dicspl.keys()):
        print("\tClass ", key)
        ltr_i = dicspl[key][0]
        lte_i = dicspl[key][1]

        if len(ltr_i) <= samples_per_class:
            thr = len(ltr_i) # Find a pivot, and we simply this chunk
        else:
            thr = samples_per_class

        ltr = ltr + ltr_i[0:thr]
        lte = lte + lte_i[0:thr]
        print("\t\t Train: {0} \t Test: {1}".format(len(ltr_i[0:thr]), len(lte[0:thr])))

    mtr = np.array(ltr)
    mte = np.array(lte)

    meta_tr = mtr[:, 0:3]
    meta_te = mte[:, 0:3]

    Ytr = mtr[:, 3]
    Yte = mte[:, 3]

    Xtr = mtr[:, 4:]
    Xte = mte[:, 4:]

    return [meta_tr, meta_te, Xtr, Ytr, Xte, Yte]



################
# Main program #
################

labels = "dbuiltup;dforest;drecreation;dbrr;dwrl;dwrn;dwrr;dcamping;dcaravan;dcross;dgolf;dheem;dhaven;dsafari;dwater;lu;lc;attr;dbath;meanhaz;stdhaz;exp".split(";")
path_in = r"/home/irene/PycharmProjects/04_Risk_Model/data/no_split/500m/nl_features_v1.12.csv"
# path_pred = r"M:\Documents\workspace\Special\IJHG\data\versions\country_features\v11\nl_centroids_v1.11.csv"
# path_out_tif = r"D:\GeoData\workspaceimg\Special\04_Risk_Model\NL_TB_Risk_v1.12.tif"

# Loading data for train/test
meta_m = np.loadtxt(path_in, delimiter=";", usecols=range(0, 3), skiprows=1)
data_m = np.loadtxt(path_in, delimiter=";", usecols=range(3, 22), skiprows=1)
Y = data_m[:, 0]
X = data_m[:, 1:]
X[np.isnan(X)] = 0

# Loading data for prediction (no target here)
ignore_target_column = 3
meta_p = np.loadtxt(path_in, delimiter=";", usecols=range(0, ignore_target_column), skiprows=1)
data_p = np.loadtxt(path_in, delimiter=";", usecols=range(ignore_target_column+1, 22), skiprows=1)
Xp = data_p
Xp[np.isnan(Xp)] = 0

# This is to filter values
# beyond sigma std. dev. from the mean
# Check outlier_detection.py for that
# stdev = 25
# Y[Y>stdev] = stdev

plt.hist(Y, bins=300)
plt.show()

print(np.unique(Y, return_counts=True))

# Processing train/test data
Ytrim, Xtrim = trim_ones(Y, X)

# Ycl = classify_target(Ynz.reshape(-1, 1))

# meta_tr, meta_te, xtrain, ytrain, xtest, ytest = shuffle_split_balance_samples(meta_m, Ycl, Xnz, samples_per_class)