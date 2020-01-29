import csv
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
from matplotlib import rcParams


poi_prop_err_singmat = []
nb_prop_err_singmat = []
zip_prop_err_singmat = []
zinb_prop_err_singmat = []

path_out_err = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/to_keep_for_taylor/long_run_4mod_v1.0_error_metrics_NESTI_5.csv"
path_out_zer = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/to_keep_for_taylor/long_run_4mod_v1.0_proportion_zeros"


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



def process_pack(key, label, dic, mod, pred, rmse):
    if mod != None:
        dic[key] = (label, mod, pred, rmse)
    else:
        dic[key] = ("Singular matrix",)
    return dic

def my_rmse(ytest, pred):
    l1, l2 = [[] for i in range(2)]
    # pos = np.where(pred!=200)[0]
    pos = np.where(pred[(pred != 200) | (pred != 0)])[0]
    print("The pos: ", pos)
    for i in pos:
        l1.append(ytest[i])
        l2.append(pred[i])
    new_ytest =  np.array(l1)
    new_pred = np.array(l2)
    rmse = np.sqrt(mean_squared_error(new_ytest, new_pred))
    return rmse

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
        print()
        print("Retrieving ensemble predicting per leaf node")
        print("Tree: {0}\t Leaf node: {1}\t Samples received: {2}\t ".format(key[0], key[1], len(dicori[key])))
        pred_tree = prediction_single_tree(ensemble, 0, dicori[key])
        if len(pred_tree) != 0:
            dicpred[key] = (pred_tree, )
        else:
            dicpred[key] = ("N/A", )
    return dicpred


def fitting_four_models_leaf_nodes(dicori):
    poi_dic, nb_dic, zip_dic, zinb_dic = [{} for i in range(4)]
    for key in sorted(dicori.keys()):
        print()
        print("Tree: {0}\t Leaf node: {1}\t Samples received: {2}\t ".format(key[0], key[1], len(dicori[key])))
        poi_mod, poi_pred, poi_rmse = tiny_poisson(dicori[key])
        nb_mod, nb_pred, nb_rmse = tiny_negbin(dicori[key])
        zip_mod, zip_pred, zip_rmse = tiny_zip(dicori[key])
        zinb_mod, zinb_pred, zinb_rmse = tiny_zinb(dicori[key])

        poi_dic = process_pack(key, "poi", poi_dic, poi_mod, poi_pred, poi_rmse)
        nb_dic = process_pack(key, "nb", nb_dic, nb_mod, nb_pred, nb_rmse)
        zip_dic = process_pack(key, "zip", zip_dic, zip_mod, zip_pred, zip_rmse)
        zinb_dic = process_pack(key, "zinb", zinb_dic, zinb_mod, zinb_pred, zinb_rmse)

    return [poi_dic, nb_dic, zip_dic, zinb_dic]



def predicting_four_models_leaf_nodes(ytest, xtest, pack):
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
        mean_pred[mean_pred>150] = 150

        rmse = np.sqrt(mean_squared_error(ytest, mean_pred))
        # rmse = my_rmse(ytest, mean_pred)
        r2 = r2_score(ytest, mean_pred)

        p.append(mean_pred)

        print("\tDoing the mean of models: ", len(l))
        print("\tDiscarded these models: ", discarded)
        print("\tSingular matrices: ", singular)
        print("\tShape of prediction: ", mean_pred.shape)
        print(mean_pred[0:30])
        print(ytest[0:30])
        print("\tRMSE: ", rmse)
        print("\tR2: ", r2)

        l = []
        discarded = 0
        singular = 0
        once = False

    return np.array(p)


def plot_compare_histograms(ax, nrow, samples_per_leaf, mean_ytest, ytest, pred, pred_rf):
    ax[nrow, 0].set_title("Poisson vs True, SPL: {0}".format(samples_per_leaf))
    ax[nrow, 0].axvline(x=mean_ytest, color="#516572", label="meanyte")
    ax[nrow, 0].hist(ytest, bins=50, label="true", color="orange")
    ax[nrow, 0].hist(pred[:, 0], bins=50, label="pred", color="#1e488f")
    ax[nrow, 0].grid()
    ax[nrow, 0].set_ylim(0, 10000)
    ax[nrow, 0].set_xlim(0, 150)
    ax[nrow, 0].legend(loc="upper right")

    ax[nrow, 1].set_title("NB vs True, SPL: {0}".format(samples_per_leaf))
    ax[nrow, 1].axvline(x=mean_ytest, color="#516572", label="meanyte")
    ax[nrow, 1].hist(ytest, bins=50, label="true", color="orange")
    ax[nrow, 1].hist(pred[:, 1], bins=50, label="pred", color="#1e488f")
    ax[nrow, 1].grid()
    ax[nrow, 1].set_ylim(0, 10000)
    ax[nrow, 1].set_xlim(0, 150)
    ax[nrow, 1].legend(loc="upper right")

    ax[nrow, 2].set_title("ZIP vs True, SPL: {0}".format(samples_per_leaf))
    ax[nrow, 2].axvline(x=mean_ytest, color="#516572", label="meanyte")
    ax[nrow, 2].hist(ytest, bins=50, label="true", color="orange")
    ax[nrow, 2].hist(pred[:, 2], bins=50, label="pred", color="#1e488f")
    ax[nrow, 2].grid()
    ax[nrow, 2].set_ylim(0, 10000)
    ax[nrow, 2].set_xlim(0, 150)
    ax[nrow, 2].legend(loc="upper right")

    ax[nrow, 3].set_title("ZINB vs True, SPL: {0}".format(samples_per_leaf))
    ax[nrow, 3].axvline(x=mean_ytest, color="#516572", label="meanyte")
    ax[nrow, 3].hist(ytest, bins=50, label="true", color="orange")
    ax[nrow, 3].hist(pred[:, 3], bins=50, label="pred", color="#1e488f")
    ax[nrow, 3].grid()
    ax[nrow, 3].set_ylim(0, 10000)
    ax[nrow, 3].set_xlim(0, 150)
    ax[nrow, 3].legend(loc="upper right")

    ax[nrow, 4].set_title("RF-Classic vs True, SPL: {0}".format(samples_per_leaf))
    ax[nrow, 4].axvline(x=mean_ytest, color="#516572", label="meanyte")
    ax[nrow, 4].hist(ytest, label="true", color="orange")
    ax[nrow, 4].hist(pred_rf, bins=50, label="pred", color="#1e488f")
    ax[nrow, 4].grid()
    ax[nrow, 4].set_ylim(0, 10000)
    ax[nrow, 4].set_xlim(0, 150)
    ax[nrow, 4].legend(loc="upper right")

    return ax



def boxplots_predictions():

    labels = ['Poisson', 'NB', 'ZIP', "ZINB", "RF-Canon"]

    path_in = r"/home/irene/PycharmProjects/04_Risk_Model/data/skewed_leaves/to_keep_for_taylor/testing_SPL{0}_NESTI{1}.csv"

    n_estimators = [1, 3, 5]
    samples_per_leaf = [200, 400, 600, 1000, 1200]

    labelsize = 16
    rcParams['xtick.labelsize'] = labelsize
    rcParams['ytick.labelsize'] = labelsize

    fig, ax = plt.subplots(nrows=len(n_estimators), ncols=5)
    # fig2, ax2 = plt.subplots(nrows=1, ncols=5, figsize=(20, 8))
    tit2 = "Predicted distributions from above in function of the number of samples per leaf node"
    plt.suptitle(tit2, size=20)

    l = []
    for i in range(len(samples_per_leaf)):
        for j in range(len(n_estimators)):
            path_cur = path_in.format(samples_per_leaf[i], n_estimators[j])
            meta_arr = np.loadtxt(path_cur, delimiter=";", usecols=range(0, 3))
            arr = np.loadtxt(path_cur, delimiter=";", usecols=range(3, 9))

            # ax2[nrow2].set_title("SPL: {0}".format(samples_per_leaf))
            # ax2[nrow2].set_facecolor('#F5F5F5')
            box = ax[j, i].boxplot(arr, patch_artist=True)
            ax[j, i].set_ylim(0, 20)
            # # ax2[nrow2].xaxis.set_ticks(labels)
            # ax2[nrow2].set_xticklabels(labels, fontsize=16, fontdict={'fontsize': 16})
            # colors = ['#A87128', '#004561', '#3C5B43', '#85243C', '#615048']
            # for patch, color in zip(box['boxes'], colors):
            #     patch.set_facecolor(color)
            #
            # ax2[nrow2].yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
            # nrow += 1
            # nrow2 += 1

    plt.show()





################
# Main program #
################

boxplots_predictions()

