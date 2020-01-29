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

poi_prop_err_singmat = []
nb_prop_err_singmat = []
zip_prop_err_singmat = []
zinb_prop_err_singmat = []

path_out_err = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/short_run_4mod_v1.0_error_metrics_NESTI_5.csv"
path_out_zer = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/short_run_4mod_v1.0_proportion_zeros"

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


# def process_pack(key, label, dic, mod, pred, rmse):
#     if mod == None:
#         dic[key] = ("Singular matrix",)
#     elif mod == "ReturnZeros":
#         dic[key] = (label, "ReturnZeros", pred, rmse)
#     else:
#         dic[key] = (label, mod, pred, rmse)
#     return dic

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
    print_exceptions(dic_pack)
    return dic_pack


def print_exceptions(pack):
    i = 0
    dicex = defaultdict()
    dicini = {"Singular":[1, 0, 0, 0, 0], "Assert":[0, 1, 0, 0, 0], "ValueError":[0, 0, 1, 0, 0], "AllZeros":[0, 0, 0, 1, 0],
              "poi":[0, 0, 0, 0, 1], "nb":[0, 0, 0, 0, 1], "zip":[0, 0, 0, 0, 1], "zinb":[0, 0, 0, 0, 1]}
    for p in pack:
        for key in sorted(p.keys()):
            newkey = (p[key][0], key[0])
            thiskey = p[key][1]

            try:
                curr = dicex[newkey]
                if thiskey == "Singular":
                    curr[0] = curr[0] + 1
                elif thiskey == "Assert":
                    curr[1] = curr[1] + 1
                elif thiskey == "ValueError":
                    curr[2] = curr[2] + 1
                elif thiskey == "AllZeros":
                    curr[3] = curr[3] + 1
                else:
                    curr[4] = curr[4] + 1
                dicex[newkey] = curr
            except ValueError:
                dicex[newkey] = dicini[thiskey]
            except KeyError:
                if type(thiskey) is not str:
                    dicex[newkey] = dicini[p[key][0]]
                else:
                    dicex[newkey] = dicini[thiskey]

            i += 1

    print("This is dicex: ", dicex)
    return dicex


##########
# Part 3 #
##########

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


def write_board(board, t, meta, pred, z=0):
    for i in range(len(meta)):
        rowid = int(meta[i][0])
        lat = meta[i][1]
        lon = meta[i][2]
        pre_i = pred[i]
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




def testing_four_models_leaf_nodes_v2(ensemble, spl, n_esti, ytest, xtest, pack, pred_rf, meta_m_test):

    singular = 0

    l, p = [[] for i in range(0, 2)]

    dicz = {"poi":0, "nb":1, "zip":2,"zinb":3}

    leaves = ensemble.apply(xtest)

    dic_test = samples_per_leaf_node_no_target(meta_m_test, leaves, xtest, ytest)

    board = np.ones((40000, n_esti)) * -99

    with open(path_out_err, "a", newline="") as w:
        writer = csv.writer(w, delimiter=";")
        for dictionary in pack:
            for key in sorted(dictionary.keys()):
                if len(dictionary[key]) == 4:
                    if len(dic_test[key]) > 0:

                        label = dictionary[key][0]
                        model = dictionary[key][1]
                        predtr = dictionary[key][2]
                        rmsetr = dictionary[key][3]

                        tiny_samples = np.array(dic_test[key])
                        xte = np.array([item[4:] for item in tiny_samples])
                        yte = np.array([item[3] for item in tiny_samples])
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

                            tiny_pred = np.zeros(yte.shape)
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
            # plt.title(label)
            # plt.hist(mean_ens, bins=20)
            # plt.show()
            # plt.clf()
            mean_ens[mean_ens>150] = 150

            # Write the results of each skewed model to the error file
            r2_sk, rmse_sk, rmsep_sk, mape_sk, smape_sk, aic_sk = calculate_error_metrics(ytest, mean_ens)
            newrow = [spl, n_esti, label, r2_sk, rmse_sk, rmsep_sk, mape_sk, smape_sk, aic_sk]
            writer.writerow(newrow)

            p.append(mean_ens)

        # Now we write the results of RF-classic
        r2_rf, rmse_rf, rmsep_rf, mape_rf, smape_rf, aic_rf = calculate_error_metrics(ytest, pred_rf)
        newrow = [spl, n_esti, "rf", r2_rf, rmse_rf, rmsep_rf, mape_rf, smape_rf, aic_rf]
        writer.writerow(newrow)

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
        pred_tree = prediction_single_tree(ensemble, 0, 2, dicori[key])
        if len(pred_tree) != 0:
            dicpred[key] = (pred_tree, )
        else:
            dicpred[key] = ("N/A", )
    return dicpred

def trim_value(Y, X, meta, v):
    pos_nze = np.where(Y!=v)
    lx, ly, lm = [[] for i in range(3)]
    for pos in pos_nze:
        ly.append(Y[pos])
        lx.append(X[pos,:])
        lm.append(meta[pos, :])
    Y_ = np.squeeze(np.array(ly), axis=0)
    X_ = np.squeeze(np.array(lx), axis=0)
    meta_ = np.squeeze(np.array(lm), axis=0)
    print(Y_.shape, X_.shape, meta_.shape)
    return [Y_, X_, meta_]



def test_varying_samples_per_node(meta_m, m):

    path_out = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/testing_varying_SPL/testing_SPL{0}_NESTI{1).csv"

    print("Type m: ", m.dtype)

    Y = m[:,0]
    X = m[:,1:]

    Ynz, Xnz, meta_m_nz = trim_value(Y, X, meta_m, 0)

    xtrain, xtest, ytrain, ytest, meta_m_train, meta_m_test = train_test_split(X, Y, meta_m, train_size=0.60, random_state=0)

    print('Type xtrain: ', xtrain.dtype)
    print("Raw data: ", Y.shape, X.shape)
    print("Train data", ytrain.shape, xtrain.shape)
    print("Test data: ", ytest.shape, xtest.shape)

    # n_estimators = [1, 5, 10, 50, 100]
    # samples_per_leaf = range(100, 1600, 100)

    n_estimators = [10]
    samples_per_leaf = [200, 400, 600, 1000, 1200]

    start_all = time.time()

    fig, ax = plt.subplots(nrows=5, ncols=5)

    nrow = 0

    for spl in samples_per_leaf:
        start_it = time.time()
        for n_esti in n_estimators:

            print()
            print("Analysis: RF with Skewed Leaves")
            print("Samples per leaf node: ", spl)
            print("Number of estimators: ", n_esti)
            print("-" * 50)

            ensemble = RandomForestRegressor(n_estimators=n_esti, min_samples_leaf=spl, bootstrap=True)

            ensemble.fit(xtrain, ytrain)

            leaves = ensemble.apply(xtrain)

            dicori = samples_per_leaf_node(leaves, xtrain, ytrain)

            pack = fitting_four_models_leaf_nodes(dicori)

            pred_rf = ensemble.predict(xtest)

            pred_sk = testing_four_models_leaf_nodes_v2(ensemble, spl, n_esti, ytest, xtest, pack, pred_rf, meta_m_test).T

            print("This is pred sk: ", pred_sk.shape)

            # stack = np.hstack((meta_m_test, pred))

            # dicens = ensemble_predictions_leaf_nodes(ensemble, dicori)

            write_proportion_of_zeros(spl, n_esti)

            # plt.subplot(2, 2, 1)
            # plt.hist(pred_sk[:, 0], bins=20)
            # plt.subplot(2, 2, 2)
            # plt.hist(pred_sk[:, 1], bins=20)
            # plt.subplot(2, 2, 3)
            # plt.hist(pred_sk[:, 2], bins=20)
            # plt.subplot(2, 2, 4)
            # plt.hist(pred_sk[:, 3], bins=20)
            # plt.show()

            # print(meta_m_test.shape, ytest.shape)
            # stack = np.hstack((meta_m, Y.reshape(-1, 1)))
            #
            # with open(path_out, "w", newline="") as w:
            #     writer = csv.writer(w, delimiter=";")
            #     for item in stack:
            #         writer.writerow(item)

            plot_compare_histograms(ax, nrow, spl, ytest, pred_sk, pred_rf)

            nrow += 1

            stop_it = time.time()
            print("--- Iteration elapsed {0} minutes ---".format(np.divide(stop_it-start_it, 60)))

    plt.show()
    end_all = time.time()
    print("--- Full program elapsed {0} hours ---".format(np.divide(end_all - start_all, 3600)))


def write_proportion_of_zeros(spl, n_esti):
    with open(path_out_zer, "a", newline="") as w:
        writer = csv.writer(w, delimiter=";")

        poi_mean = np.round(np.mean(poi_prop_err_singmat), decimals=0)
        nb_mean = np.round(np.mean(nb_prop_err_singmat), decimals=0)
        zip_mean = np.round(np.mean(zip_prop_err_singmat), decimals=0)
        zinb_mean = np.round(np.mean(zinb_prop_err_singmat), decimals=0)

        poi_len =len(poi_prop_err_singmat)
        nb_len = len(nb_prop_err_singmat)
        zip_len = len(zip_prop_err_singmat)
        zinb_len = len(zinb_prop_err_singmat)

        global poi_prop_err_singmat
        global nb_prop_err_singmat
        global zip_prop_err_singmat
        global zinb_prop_err_singmat

        poi_prop_err_singmat = []
        nb_prop_err_singmat = []
        zip_prop_err_singmat = []
        zinb_prop_err_singmat = []

        newrow = [spl, n_esti, poi_mean, nb_mean, zip_mean, zinb_mean, poi_len, nb_len, zip_len, zinb_len]
        writer.writerow(newrow)


def draw_boxplot():
    labels = ["Poisson", "NB", "ZIP", "ZINB"]

    all_data = [poi_prop_err_singmat, nb_prop_err_singmat, zip_prop_err_singmat, zinb_prop_err_singmat]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))

    # rectangular box plot
    bplot1 = ax.boxplot(all_data, vert=True, patch_artist=True, labels=labels)  # will be used to label x-ticks
    ax.set_title('Rectangular box plot')

    # fill with colors
    colors = ['darkslateblue', 'darkslateblue', 'slategray', "indianred"]
    for bplot in (bplot1, ):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    # adding horizontal grid lines
    ax.yaxis.grid(True)
    ax.set_xlabel('Three separate samples')
    ax.set_ylabel('Observed values')

    plt.show()

def plot_compare_histograms(ax, nrow, samples_per_leaf, ytest, pred_sk, pred_rf):

    titles_list = ["Poisson vs True, SPL: {0}", "NB vs True, SPL: {0}",
                   "ZIP vs True, SPL: {0}", "ZINB vs True, SPL: {0}",
                   "RF-Classic vs True, SPL: {0}"]

    mean_ytest = np.mean(ytest)

    # print()
    # print()
    # print("Poisson")
    # print(np.unique(pred_sk[:, 0], return_counts=True))
    # print("nb")
    # print(np.unique(pred_sk[:, 1], return_counts=True))
    # print("zip")
    # print(np.unique(pred_sk[:, 2], return_counts=True))
    # print("zinb")
    # print(np.unique(pred_sk[:, 3], return_counts=True))
    # print()
    # print()
    # print("This is nrow: ", nrow)

    for j in range(0, 5):

        ax[nrow, j].hist(ytest, bins=80, label="Target", color="orange")

        if j == 4:

            print(np.unique(pred_rf, return_counts=True))

            ax[nrow, 4].hist(pred_rf, bins=60, label="pred. rf", color="#1e488f")
        else:

            ax[nrow, j].hist(pred_sk[:, j], bins=60, label="pred. skew", color="#1e488f")

        ax[nrow, j].set_title(titles_list[j].format(samples_per_leaf))

        ax[nrow, j].axvline(x=mean_ytest, color="#516572", label="Mean target")

        ax[nrow, j].set_ylim(0, 10000)

        ax[nrow, j].set_xlim(0, 60)

        ax[nrow, j].legend(loc="upper right")

        ax[nrow, j].grid()

    return ax



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

all_preds = test_varying_samples_per_node(meta_m, m)

