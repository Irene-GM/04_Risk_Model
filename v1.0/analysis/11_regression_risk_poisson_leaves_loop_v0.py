from __future__ import print_function
import csv
import time
import numpy as np
from sklearn.tree import _tree, export_graphviz
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals.six import StringIO
import pydotplus
import itertools

def process_pack(key, label, dic, mod, pred, rmse):
    if mod != None:
        dic[key] = (label, mod, pred, rmse)
    else:
        dic[key] = ("Singular matrix",)
    return dic

def my_rmse(ytest, pred):
    l1, l2 = [[] for i in range(2)]
    pos = np.where(pred!=-2)[0]
    # pos = np.where(pred[(pred != 200) | (pred != 0)])[0]
    for i in pos:
        l1.append(ytest[i])
        l2.append(pred[i])
    new_ytest =  np.array(l1)
    new_pred = np.array(l2)
    rmse = np.sqrt(mean_squared_error(new_ytest, new_pred))
    r2 = np.round(r2_score(new_ytest, new_pred), decimals=4)
    return [rmse, r2]

def tiny_poisson(l):
    # print("\t\tRunning Poisson")
    poi_mod, poi_ppf_obs = [None for i in range(2)]
    poi_rmse = 0
    xtr = np.array([item[1:] for item in l])
    ytr = np.array([item[0] for item in l]).reshape(-1, 1)
    try:
        poi_mod = sm.Poisson(ytr, xtr).fit(method="newton", maxiter=50, disp=0)
        poi_mean_pred = poi_mod.predict(xtr)
        poi_ppf_obs = stats.poisson.ppf(q=0.95, mu=poi_mean_pred)  # average over x in sample
        poi_rmse = np.sqrt(mean_squared_error(ytr, poi_ppf_obs))

    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            pass
            # print("\t\t\tIgnored a singular matrix.")
    except ValueError:
        pass
        # print("\t\t\tIgnored output containing np.nan or np.inf")

    return [poi_mod, poi_ppf_obs, poi_rmse]


def tiny_negbin(l):
    # print("\t\tRunning NegBin")
    nb_mod, nb_pred = [None for i in range(2)]
    nb_rmse = 0
    xtr = np.array([item[1:] for item in l])
    ytr = np.array([item[0] for item in l]).reshape(-1, 1)
    try:
        nb_mod = sm.NegativeBinomial(ytr, xtr).fit(start_params=None, method='newton', maxiter=50, disp=0)
        nb_pred = nb_mod.predict(xtr)
        nb_rmse = np.sqrt(mean_squared_error(ytr, nb_pred))
    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            pass
            # print("\t\t\tIgnored a singular matrix.")
    except ValueError:
        pass
        # print("\t\t\tIgnored output containing np.nan or np.inf")

    return [nb_mod, nb_pred, nb_rmse]


def tiny_zip(l):
    # print("\t\tRunning Zero-Inflated Poisson")
    zip_mod, zip_ppf_obs, zip_pred = [None for i in range(3)]
    zip_rmse = 0
    xtr = np.array([item[1:] for item in l])
    ytr = np.array([item[0] for item in l]).reshape(-1, 1)
    try:
        zip_mod = sm.ZeroInflatedPoisson(ytr, xtr).fit(method="newton", maxiter=50, disp=0)
        zip_mean_pred = zip_mod.predict(xtr, exog_infl=np.ones((len(xtr), 1)))
        zip_ppf_obs = stats.poisson.ppf(q=0.95, mu=zip_mean_pred)
        zip_rmse = np.sqrt(mean_squared_error(ytr, zip_ppf_obs))
    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            pass
            # print("\t\t\tIgnored a singular matrix.")
    except ValueError:
        pass
        # print("\t\t\tIgnored output containing np.nan or np.inf")
    return [zip_mod, zip_ppf_obs, zip_rmse]

def tiny_zinb(l):
    # print("\t\tRunning Zero-Inflated NegBin")
    zinb_mod, zinb_pred = [None for i in range(2)]
    zinb_rmse = 0

    xtr = np.array([item[1:] for item in l])
    ytr = np.array([item[0] for item in l]).reshape(-1, 1)

    try:
        zinb_mod = sm.ZeroInflatedNegativeBinomialP(ytr, xtr).fit(method="newton", maxiter=50, disp=0)
        zinb_pred = zinb_mod.predict(xtr, exog_infl=np.ones((len(xtr), 1)))
        zinb_rmse = np.sqrt(mean_squared_error(ytr, zinb_pred))

    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            pass
            # print("\t\t\tIgnored a singular matrix.")
    except ValueError:
        pass
        # print("\t\t\tIgnored output containing np.nan or np.inf")
    return [zinb_mod, zinb_pred, zinb_rmse]


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

def prediction_single_tree(ensemble, t, l):
    xtr = np.array([item[1:] for item in l])
    ytr = np.array([item[0] for item in l]).reshape(-1, 1)
    tree = ensemble.estimators_[t].tree_
    pred_tree = tree.predict(xtr.astype(np.float32))
    apply_tree = tree.apply(xtr.astype(np.float32))
    # draw_tree(ensemble, t)
    # save_tree_graph(ensemble, t)
    return pred_tree

def ensemble_predictions_leaf_nodes(ensemble, dicori):
    dicpred = defaultdict(tuple)
    for key in sorted(dicori.keys()):
        # print()
        # print("Retrieving ensemble predicting per leaf node")
        # print("Tree: {0}\t Leaf node: {1}\t Samples received: {2}\t ".format(key[0], key[1], len(dicori[key])))
        pred_tree = prediction_single_tree(ensemble, 0, dicori[key])
        if len(pred_tree) != 0:
            dicpred[key] = (pred_tree, )
        else:
            dicpred[key] = ("N/A", )
    return dicpred


def fitting_four_models_leaf_nodes(dicori):
    poi_dic, nb_dic, zip_dic, zinb_dic = [{} for i in range(4)]
    for key in sorted(dicori.keys()):
        # print()
        # print("Tree: {0}\t Leaf node: {1}\t Samples received: {2}\t ".format(key[0], key[1], len(dicori[key])))
        poi_mod, poi_pred, poi_rmse = tiny_poisson(dicori[key])
        nb_mod, nb_pred, nb_rmse = tiny_negbin(dicori[key])
        zip_mod, zip_pred, zip_rmse = tiny_zip(dicori[key])
        zinb_mod, zinb_pred, zinb_rmse = tiny_zinb(dicori[key])

        poi_dic = process_pack(key, "poi", poi_dic, poi_mod, poi_pred, poi_rmse)
        nb_dic = process_pack(key, "nb", nb_dic, nb_mod, nb_pred, nb_rmse)
        zip_dic = process_pack(key, "zip", zip_dic, zip_mod, zip_pred, zip_rmse)
        zinb_dic = process_pack(key, "zinb", zinb_dic, zinb_mod, zinb_pred, zinb_rmse)

    return [poi_dic, nb_dic, zip_dic, zinb_dic]



def predicting_four_models_leaf_nodes(spl, n_esti, ytest, xtest, pack, pred_rf):
    path_err = r"/home/irene/PycharmProjects/04_Risk_Model/data/poisson_leaves/RF_skewed_leaves_error_metrics_with_zeros_v1.csv"
    with open(path_err, "a") as w:
        writer = csv.writer(w, delimiter=";")
        l, p = [[] for i in range(0, 2)]
        discarded, singular = [0 for i in range(0, 2)]
        once = False
        for dictionary in pack:
            for key in sorted(dictionary.keys()):
                if len(dictionary[key]) == 4:
                    label = dictionary[key][0]
                    model = dictionary[key][1]
                    predtr = dictionary[key][2]
                    rmsetr = dictionary[key][3]
                    if once == False:
                        print("\nPredicting for: ", label)
                        once = True

                    if label in ["zip", "zinb"]:
                        predte = model.predict(xtest, exog_infl=np.ones((len(xtest), 1)))
                    else:
                        predte = model.predict(xtest)

                    if not np.isnan(predte).all():
                        l.append(predte)
                    else:
                        discarded += 1
                else:
                    singular += 1

            mean_pred  = np.round(np.mean(np.array(l), axis=0), decimals=0)
            mean_pred[np.isnan(mean_pred)] = -2
            mean_pred[mean_pred>200] = 200

            rmse = np.round(np.sqrt(mean_squared_error(ytest, mean_pred)), decimals=4)
            myrmse, myr2 = my_rmse(ytest, mean_pred)

            p.append(mean_pred)

            rmse_rf = np.sqrt(mean_squared_error(ytest, pred_rf))
            r2_rf = r2_score(ytest, pred_rf)

            newrow = [spl, n_esti, label, rmse, myrmse, rmse_rf, myr2, r2_rf]
            writer.writerow(newrow)

            l = []
            discarded = 0
            singular = 0
            once = False

    return np.array(p)


def test_varying_samples_per_node(meta_m, m):
    print("Type m: ", m.dtype)
    path_out = r"/home/irene/PycharmProjects/04_Risk_Model/data/poisson_leaves/prediction_ytest.csv"

    Y = m[:,0]
    X = m[:,1:]

    # Ynz, Xnz = trim_value(Y, X, 0)

    xtrain, xtest, ytrain, ytest, meta_m_train, meta_m_test = train_test_split(X, Y, meta_m, train_size=0.60, random_state=42)

    print('Type xtrain: ', xtrain.dtype)
    print("Raw data: ", Y.shape, X.shape)
    print("Train data", ytrain.shape, xtrain.shape)
    print("Test data: ", ytest.shape, xtest.shape)

    n_estimators = [1, 5, 10, 50, 100]
    samples_per_leaf = range(100, 1600, 100)

    start_all = time.time()
    for spl in samples_per_leaf:
        start_it = time.time()
        for n_esti in n_estimators:

            print()
            print("Analysis: RF with Poisson Leaves")
            print("Samples per leaf node: ", spl)
            print("Number of estimators: ", n_esti)
            print("-" * 50)

            ensemble = RandomForestRegressor(n_estimators=n_esti, min_samples_leaf=spl, bootstrap=False)

            ensemble.fit(xtrain, ytrain)

            leaves = ensemble.apply(xtrain)

            dicori = samples_per_leaf_node(leaves, xtrain, ytrain)

            pack = fitting_four_models_leaf_nodes(dicori)

            pred_rf = ensemble.predict(xtest)

            pred = predicting_four_models_leaf_nodes(spl, n_esti, ytest, xtest, pack, pred_rf).T

            stack = np.hstack((meta_m_test, pred))

            dicens = ensemble_predictions_leaf_nodes(ensemble, dicori)

            stop_it = time.time()
            print("--- Iteration elapsed {0} minutes ---".format(np.divide(stop_it-start_it, 60)))

    end_all = time.time()
    print("--- Full program elapsed {0} hours ---".format(np.divide(end_all - start_all, 3600)))








################
# Main program #
################

path_in = r"/media/irene/MyPassport/acies/workspace/Special/IJHG/data/versions/country_features/v12/nl_risk_features_v1.12.csv"
path_nl_500m = r"/home/irene/PycharmProjects/04_Risk_Model/data/no_split/500m/nl_centroids_500m_v1.12.csv"

metacols = range(0, 3)
datacols = range(3, 25)
predcols = range(3, 22)

meta_m = np.loadtxt(path_in, delimiter=";", usecols=metacols, skiprows=1)
m = np.loadtxt(path_in, delimiter=";", usecols=datacols, skiprows=1, dtype=np.float32)

meta_p = np.loadtxt(path_nl_500m, delimiter=";", usecols=metacols, skiprows=1)
p = np.loadtxt(path_nl_500m, delimiter=";", usecols=predcols, skiprows=1, dtype=np.float32)


all_preds = test_varying_samples_per_node(meta_m, m)

