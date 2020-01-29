import csv
import time
import pylab
import pydotplus
import itertools
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.externals.six import StringIO
from sklearn.tree import _tree, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import normaltest, skew, kurtosis, boxcox

import warnings
from statsmodels.tools.sm_exceptions import HessianInversionWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=HessianInversionWarning)


poi_prop_err_singmat = []
nb_prop_err_singmat = []
zip_prop_err_singmat = []
zinb_prop_err_singmat = []

##########
# Part 1 #
##########

def samples_per_leaf_node(leaves, xs, ys):
    t = 0
    dic = defaultdict(list)
    for tree_node_list in leaves.T:
        j = 0
        for node in tree_node_list:
            key = (t, node)
            row_j = [ys[j]] + xs[j, :].tolist()
            dic[key].append(row_j)
            j += 1
        t += 1
    return dic

def samples_per_leaf_node_no_target(meta, leaves, xs, ys):
    t = 0
    dic = defaultdict(list)
    for tree_node_list in leaves.T:
        j = 0
        for node in tree_node_list:
            key = (t, node)
            row_j = meta[j,:].tolist() + [ys[j]] + xs[j, :].tolist()
            dic[key].append(row_j)
            j += 1
        t += 1
    return dic


def samples_per_leaf_node_NL(meta, leaves, xs):
    t = 0
    dic = defaultdict(list)
    for tree_node_list in leaves.T:
        j = 0
        for node in tree_node_list:
            key = (t, node)
            row_j = meta[j,:].tolist() + xs[j, :].tolist()
            dic[key].append(row_j)
            j += 1
        t += 1
    return dic




##########
# Part 2 #
##########

def return_zeros(ytr, e):
    if e == "Singular":
        mod = "Singular"
    elif e == "Assert":
        mod = "Assert"
    elif e == "ValueError":
        mod = "ValueError"
    elif e == "AllZeros":
        mod = "AllZeros"

    pred = np.zeros(ytr.shape)
    rmse_tr = 0
    res = [mod, pred, rmse_tr]
    return res

def process_pack(key, label, dic, mod, pred, rmse):
    dic[key] = (label, mod, pred, rmse)
    return dic


def tiny_poisson(l):
    poi_mod, poi_ppf_obs = [None for i in range(2)]
    poi_rmse = 0
    xtr = np.array([item[1:] for item in l])
    ytr = np.array([item[0] for item in l]).reshape(-1, 1)

    poi_res = []
    try:
        if np.count_nonzero(ytr) > 0:
            poi_mod = sm.Poisson(ytr, xtr).fit(method="nm", maxiter=10000, disp=0, maxfun=10000) # method nm works without singular mat
            poi_mean_pred = poi_mod.predict(xtr)
            poi_ppf_obs = stats.poisson.ppf(q=0.99, mu=poi_mean_pred)
            poi_ppf_obs[poi_ppf_obs>150] = 150
            poi_rmse_tr = np.sqrt(mean_squared_error(ytr, poi_ppf_obs))
            poi_res = [poi_mod, poi_ppf_obs, poi_rmse_tr]

        else:
            poi_res = return_zeros(ytr, "AllZeros")

    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            # print(" You should not have reached this point. ")
            # print(" Regularization should avoid the singular matrix. ")
            nzeros = len(ytr) - np.count_nonzero(ytr)
            prop = round((100 * nzeros) / len(ytr), 2)
            # print(" Proportion of zeros: ", prop)
            poi_prop_err_singmat.append(prop)
            nb_res = return_zeros(ytr, "Singular")
    except AssertionError as e:
        nb_res = return_zeros(ytr, "Assert")
    except ValueError as e:
        print("\t\t\tIgnored output containing np.nan or np.inf")
        pass
    return poi_res


def tiny_negbin(l):
    nb_mod, nb_pred = [None for i in range(2)]
    nb_rmse = 0
    xtr = np.array([item[1:] for item in l])
    ytr = np.array([item[0] for item in l]).reshape(-1, 1)
    nb_res = []

    try:
        if np.count_nonzero(ytr) > 0:
            nb_mod = sm.NegativeBinomialP(ytr, xtr).fit_regularized(maxiter=10000, disp=0, maxfun=10000, exposure=None, offset=None)
            # print(nb_mod.summary())
            nb_mean_pred = nb_mod.predict(xtr, which="mean")
            nb_rmse_tr = np.sqrt(mean_squared_error(ytr, nb_mean_pred))
            nb_res = [nb_mod, nb_mean_pred, nb_rmse_tr]
        else:
            nb_res = return_zeros(ytr, "AllZeros")

    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            # print(" You should not have reached this point. ")
            # print(" Regularization should avoid the singular matrix. ")
            nzeros = len(ytr) - np.count_nonzero(ytr)
            prop = round((100 * nzeros) / len(ytr), 2)
            # print(" Proportion of zeros: ", prop)
            nb_res = return_zeros(ytr, "Singular")
            nb_prop_err_singmat.append(prop)

    except AssertionError as e:
        nb_res = return_zeros(ytr, "Assert")
    except ValueError as e:
        print("\t\t\tIgnored output containing np.nan or np.inf")
        pass
    return nb_res


def tiny_zip(l):
    zip_mod, zip_ppf_obs, zip_pred = [None for i in range(3)]
    zip_rmse = 0
    xtr = np.array([item[1:] for item in l])
    ytr = np.array([item[0] for item in l]).reshape(-1, 1)
    zip_res = []
    try:
        if np.count_nonzero(ytr) > 0:
            zip_mod = sm.ZeroInflatedPoisson(ytr, xtr).fit_regularized(maxiter=10000, disp=0, maxfun=10000)
            # print(zip_mod.summary())
            zip_mean_pred = zip_mod.predict(xtr, exog_infl=np.ones((len(xtr), 1)))
            zip_ppf_obs = stats.poisson.ppf(q=0.95, mu=zip_mean_pred)
            zip_rmse_tr = np.sqrt(mean_squared_error(ytr, zip_ppf_obs))
            zip_res = [zip_mod, zip_ppf_obs, zip_rmse_tr]
        else:
            zip_res = return_zeros(ytr, "AllZeros")

    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            # print(" You should not have reached this point. ")
            # print(" Regularization should avoid the singular matrix. ")
            nzeros = len(ytr) - np.count_nonzero(ytr)
            zip_res = return_zeros(ytr, "Singular")
            prop = round((100 * nzeros) / len(ytr), 2)
            # print(" Proportion of zeros: ", prop)
            zip_prop_err_singmat.append(prop)
    except AssertionError as e:
        zip_res = return_zeros(ytr, "Assert")
    except ValueError as e:
        print("\t\t\tIgnored output containing np.nan or np.inf")
        pass
    return zip_res

def tiny_zinb(l):
    zinb_mod, zinb_pred = [None for i in range(2)]
    zinb_rmse = 0

    xtr = np.array([item[1:] for item in l])
    ytr = np.array([item[0] for item in l]).reshape(-1, 1)

    try:
        if np.count_nonzero(ytr) > 0:
            zinb_mod = sm.ZeroInflatedNegativeBinomialP(ytr, xtr).fit_regularized(maxiter=10000, disp=0, maxfun=10000) #nm va
            # print(zinb_mod.summary())
            zinb_pred = zinb_mod.predict(xtr, which="mean", exog_infl=np.ones((len(xtr), 1)))
            zinb_rmse = np.sqrt(mean_squared_error(ytr, zinb_pred))
            zinb_res = [zinb_mod, zinb_pred, zinb_rmse]
        else:
            zinb_res = return_zeros(ytr, "AllZeros")

    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            # print(" You should not have reached this point. ")
            # print(" Regularization should avoid the singular matrix. ")
            nzeros = len(ytr) - np.count_nonzero(ytr)
            zinb_res = return_zeros(ytr, "Singular")
            prop = round((100 * nzeros) / len(ytr), 2)
            # print(" Proportion of zeros: ", prop)
            zinb_prop_err_singmat.append(prop)
    except AssertionError as e:
        zinb_res = return_zeros(ytr, "Assert")
    except ValueError as e:
        print("\t\t\tIgnored output containing np.nan or np.inf")
        pass
    return zinb_res


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


    dic_pack = [poi_dic, nb_dic, zip_dic, zinb_dic]
    return dic_pack



##########
# Part 3 #
##########

def write_board(board, t, meta, pred, z=0):
    print("Board size: ", board.shape)
    for i in range(len(meta)):
        rowid = int(meta[i][0])
        lat = meta[i][1]
        lon = meta[i][2]
        pre_i = pred[i]
        print(rowid, t)
        board[rowid, t] = pre_i
    return board

def trim_board(board):
    l = []
    for i in range(len(board)):
        if -99 not in board[i,:]:
            l.append(board[i,:])
    return np.array(l)


def calculate_error_metrics(Y, pred):
    # Number of variables
    k = 21
    n = len(Y)

    resid = Y - pred
    resid[resid <= 0] = 0.5
    sse = sum(resid ** 2)

    Yperturbed = Y + np.random.normal(0.10, 0.11, Y.shape[0])

    r2 = np.round(r2_score(Y, pred), decimals=4)
    rmse = np.round(np.sqrt(mean_squared_error(Y, pred)), decimals=4)
    rmsep = 100 * np.sqrt(np.divide(sse, n)) / np.mean(Y)
    mape = np.divide(sum(np.abs(resid) / np.abs(Yperturbed)), len(Y))
    smape = np.divide(sum(2 * np.abs(resid) / (np.abs(Yperturbed) + np.abs(pred))), len(Y))
    aic = 2 * k - 2 * np.log(sse)

    return [r2, rmse, rmsep, mape, smape, aic]


def run_models_and_update_board(model, label, meta_xte, xte, board, key):
    if label == "poi":
        mean_pred = model.predict(xte)
        tiny_pred = stats.poisson.ppf(q=0.95, mu=mean_pred)
        tiny_pred[np.isnan(tiny_pred)] = 0
        board = write_board(board, key[0], meta_xte, tiny_pred, z=0)

    elif label == "nb":
        tiny_pred = model.predict(xte)
        tiny_pred[np.isnan(tiny_pred)] = 0
        board = write_board(board, key[0], meta_xte, tiny_pred, z=1)

    elif label == "zip":
        mean_pred = model.predict(xte, exog_infl=np.ones((len(xte), 1)))
        tiny_pred = stats.poisson.ppf(q=0.95, mu=mean_pred)
        tiny_pred[np.isnan(tiny_pred)] = 0
        board = write_board(board, key[0], meta_xte, tiny_pred, z=2)

    elif label == "zinb":
        tiny_pred = model.predict(xte, exog_infl=np.ones((len(xte), 1)))
        tiny_pred[np.isnan(tiny_pred)] = 0
        board = write_board(board, key[0], meta_xte, tiny_pred, z=3)

    return [tiny_pred, board]




def predicting_four_models_leaf_nodes_NL(ensemble, meta_p, the_p, pack):

    singular = 0

    l, p = [[] for i in range(0, 2)]

    dicz = {"poi":0, "nb":1, "zip":2,"zinb":3}

    xtest = np.copy(the_p)

    leaves = ensemble.apply(xtest)

    dic_test = samples_per_leaf_node_NL(meta_p, leaves, xtest)

    board = np.ones((40000, 20)) * -99

    for dictionary in pack:
        for key in sorted(dictionary.keys()):
            if len(dictionary[key]) == 4:
                if len(dic_test[key]) > 0:

                    label = dictionary[key][0]
                    model = dictionary[key][1]
                    predtr = dictionary[key][2]
                    rmsetr = dictionary[key][3]

                    tiny_samples = np.array(dic_test[key])
                    xte = np.array([item[3:] for item in tiny_samples])
                    # yte = No Yte in predicting phase
                    meta_xte = np.array([item[0:3] for item in tiny_samples])

                    if model not in ["Singular", "Assert", "ValueError", "AllZeros"]:
                        tiny_pred, board = run_models_and_update_board(model, label, meta_xte, xte, board, key)
                        if not np.isnan(tiny_pred).all():
                            l.append(tiny_pred)
                        else:
                            discarded += 1
                    else:

                        # (CONTINUE FROM NOTE 1)
                        # At this time we have encountered samples falling in a leaf node for which
                        # there is no model available due to have only received zeros.
                        # Thus we apply the same principle of RF and we propagate the zero that
                        # will be averaged outside this loop

                        tiny_pred = np.zeros((xte.shape[0], 1))
                        board = write_board(board, key[0], meta_xte, tiny_pred, z=dicz[label])

            else:
                print("This key not working: ", key, len(dic_test[key]))
                singular += 1

            l = []
            discarded = 0
            singular = 0

        # Now we compact and average the predictions
        # and also remove possible nans due to
        # numerical instability of any of the models
        trimmed = trim_board(board)
        mean_ens = np.round(np.mean(trimmed, axis=1), decimals=0)
        mean_ens[mean_ens>150] = 150
        p.append(mean_ens)

    return np.array(p)


def prediction_single_tree(ensemble, t1, t2, l):
    xtr = np.array([item[1:] for item in l])
    ytr = np.array([item[0] for item in l]).reshape(-1, 1)
    tree1 = ensemble.estimators_[t1].tree_
    pred_tree1 = tree1.predict(xtr.astype(np.float32))
    apply_tree1 = tree1.apply(xtr.astype(np.float32))

    tree2 = ensemble.estimators_[t2].tree_
    pred_tree2 = tree2.predict(xtr.astype(np.float32))
    apply_tree2 = tree2.apply(xtr.astype(np.float32))

    # save_tree_graph(ensemble, t1, t2)
    return pred_tree1

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


def predict_models(meta_m, m, meta_p, p):
    print("Type m: ", m.dtype)
    path_out = r"D:/UTwente/PycharmProjects/04_Risk_Model/data/skewed_leaves/prediction_nl_four_models_v3_20T_200S.csv"

    Y = m[:, 0]
    X = m[:, 1:]

    n_esti = 20
    spl = 200

    xtrain, xtest, ytrain, ytest, meta_m_train, meta_m_test = train_test_split(X, Y, meta_m, train_size=0.60, random_state=0)

    ensemble = RandomForestRegressor(n_estimators=n_esti, min_samples_leaf=spl, bootstrap=False)

    ensemble.fit(xtrain, ytrain)

    leaves = ensemble.apply(xtrain)

    dicori = samples_per_leaf_node(leaves, xtrain, ytrain)

    pack = fitting_four_models_leaf_nodes(dicori)

    print("Predicting with random forest")
    pred_rf = ensemble.predict(p).reshape(-1, 1)

    print("Prediction with the four models")

    pred_sk = predicting_four_models_leaf_nodes_NL(ensemble, meta_p, p, pack).T

    print(meta_p.shape, pred_sk.shape, pred_rf.shape)

    stack = np.hstack((meta_p, pred_sk, pred_rf))

    print("Stacked predictions: ", stack.shape, meta_p.shape)

    # dicens = ensemble_predictions_leaf_nodes(ensemble, dicori)

    # print(meta_m_test.shape, ytest.shape)
    # stack = np.hstack((meta_m, Y.reshape(-1, 1)))

    with open(path_out, "w", newline="") as w:
        writer = csv.writer(w, delimiter=";")
        for item in stack:
            writer.writerow(item)


################
# Main program #
################

path_in = r"F:/acies/workspace/Special/IJHG/data/versions/country_features/v12/nl_risk_features_v1.12.csv"
path_nl_500m = r"F:/acies/workspace/Special/IJHG/data/versions/country_features/v12/nl_centroids_v1.12.csv"

metacols = range(0, 3)
datacols = range(3, 25)
predcols = range(3, 24)

meta_m = np.loadtxt(path_in, delimiter=";", usecols=metacols, skiprows=1)
m = np.loadtxt(path_in, delimiter=";", usecols=datacols, skiprows=1, dtype=np.float32)

print(meta_m.shape, m.shape)

meta_p = np.loadtxt(path_nl_500m, delimiter=";", usecols=metacols, skiprows=1)
p = np.loadtxt(path_nl_500m, delimiter=";", usecols=predcols, skiprows=1, dtype=np.float32)

print(meta_p.shape, p.shape)

# all_preds = test_varying_samples_per_node(meta_m, m)

predict_models(meta_m, m, meta_p, p)
