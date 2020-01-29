import csv
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from collections import defaultdict


def samples_per_leaf_node(leaves, xtrain, ytrain):
    t = 0
    dic = defaultdict(list)
    for tree_node_list in leaves.T:
        j = 0
        for node in tree_node_list:
            key = (t, node)
            row_j = [ytrain[j]] + xtrain[j, :].tolist()
            dic[key].append(row_j)
            j += 1
        t += 1
    return dic

def samples_per_leaf_node_ensemble(meta_m, m):

    path_out_tr = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/samples_leaf_node_ensemble_train.csv"
    path_out_te = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/samples_leaf_node_ensemble_test.csv"

    Y = m[:, 0]
    X = m[:, 1:]

    xtrain, xtest, ytrain, ytest, meta_m_train, meta_m_test = train_test_split(X, Y, meta_m, train_size=0.60,
                                                                               random_state=0)

    print('Type xtrain: ', xtrain.dtype)
    print("Raw data: ", Y.shape, X.shape)
    print("Train data", ytrain.shape, xtrain.shape)
    print("Test data: ", ytest.shape, xtest.shape)

    n_estimators = [50]
    samples_per_leaf = [1000]

    header = ["ntree", "nnode", "tb_per_px"]
    with open(path_out_tr, "w", newline="") as wtr:
        with open(path_out_te, "w", newline="") as wte:
            writer_tr = csv.writer(wtr, delimiter=";")
            writer_tr.writerow(header)
            writer_te = csv.writer(wte, delimiter=";")
            writer_te.writerow(header)

            for spl in samples_per_leaf:
                for n_esti in n_estimators:
                    print()
                    print("Analysis: RF with Skewed Leaves")
                    print("Samples per leaf node: ", spl)
                    print("Number of estimators: ", n_esti)
                    print("-" * 50)

                    ensemble = RandomForestRegressor(n_estimators=n_esti, min_samples_leaf=spl, bootstrap=True)
                    ensemble.fit(xtrain, ytrain)
                    leaves_train = ensemble.apply(xtrain)
                    dicori_train = samples_per_leaf_node(leaves_train, xtrain, ytrain)

                    l = []
                    for key in sorted(dicori_train.keys()):
                        for sam in dicori_train[key]:
                            l.append(sam[0])
                        newrow = [key[0], key[1]] + l
                        writer_tr.writerow(newrow)
                        l = []

                    leaves_test = ensemble.apply(xtest)
                    dicori_test = samples_per_leaf_node(leaves_test, xtest, ytest)

                    l = []
                    for key in sorted(dicori_test.keys()):
                        for sam in dicori_test[key]:
                            l.append(sam[0])
                        newrow = [key[0], key[1]] + l
                        writer_te.writerow(newrow)
                        l = []



################
# Main program #
################

path_in = r"/media/irene/MyPassport/acies/workspace/Special/IJHG/data/versions/country_features/v12/nl_risk_features_v1.12.csv"
path_nl_500m = r"/media/irene/MyPassport/acies/workspace/Special/IJHG/data/versions/country_features/v12/nl_centroids_v1.12.csv"

metacols = range(0, 3)
datacols = range(3, 25)
predcols = range(3, 24)

meta_m = np.loadtxt(path_in, delimiter=";", usecols=metacols, skiprows=1)
m = np.loadtxt(path_in, delimiter=";", usecols=datacols, skiprows=1, dtype=np.float32)

print(meta_m.shape, m.shape)

meta_p = np.loadtxt(path_nl_500m, delimiter=";", usecols=metacols, skiprows=1)
p = np.loadtxt(path_nl_500m, delimiter=";", usecols=predcols, skiprows=1, dtype=np.float32)

print(meta_p.shape, p.shape)

all_preds = samples_per_leaf_node_ensemble(meta_m, m)