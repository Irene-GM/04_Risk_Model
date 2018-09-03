from __future__ import print_function
import numpy as np
from sklearn.tree import _tree, export_graphviz
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from sklearn.externals.six import StringIO
import pydotplus
import itertools



def leaf_depths(tree, node_id=0):
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]
    if left_child == _tree.TREE_LEAF:
        depths = np.array([0])
    else:
        left_depths = leaf_depths(tree, left_child) + 1
        right_depths = leaf_depths(tree, right_child) + 1
        depths = np.append(left_depths, right_depths)
    return depths


def leaf_samples(tree, node_id=0):
    left_child = tree.children_left[node_id]
    right_child = tree.children_right[node_id]
    if left_child == _tree.TREE_LEAF:
        samples = np.array([tree.n_node_samples[node_id]])
    else:
        left_samples = leaf_samples(tree, left_child)
        right_samples = leaf_samples(tree, right_child)
        samples = np.append(left_samples, right_samples)
    return samples


def draw_tree(ensemble, tree_id):
    plt.figure(figsize=(8, 8))
    plt.subplot(211)
    tree = ensemble.estimators_[tree_id].tree_
    depths = leaf_depths(tree)
    plt.hist(depths, histtype='step', color='#9933ff', bins=range(min(depths), max(depths) + 1))
    plt.xlabel("Depth of leaf nodes (tree %s)" % tree_id)
    plt.subplot(212)
    samples = leaf_samples(tree)
    plt.hist(samples, histtype='step', color='#3399ff', bins=range(min(samples), max(samples) + 1))
    plt.xlabel("Number of samples in leaf nodes (tree %s)" % tree_id)
    plt.show()

def trim_value(Y, X, v):
    pos_nze = np.where(Y!=v)
    lx, ly = [[] for i in range(2)]
    for pos in pos_nze:
        ly.append(Y[pos])
        lx.append(X[pos,:])
    Y_ = np.squeeze(np.array(ly), axis=0)
    X_ = np.squeeze(np.array(lx), axis=0)
    print(Y_.shape, X_.shape)
    return [Y_, X_]

def save_tree_graph(ensemble, t):
    outfile = r"/home/irene/PycharmProjects/04_Risk_Model/data/tree_plotted.png"
    dotfile = StringIO()
    export_graphviz(ensemble.estimators_[t], out_file=dotfile)
    graph = pydotplus.graph_from_dot_data(dotfile.getvalue())
    graph.write_png(outfile)

def prediction_single_tree(ensemble, t, l):
    xtr = np.array([item[1:] for item in l])
    ytr = np.array([item[0] for item in l]).reshape(-1, 1)
    tree = ensemble.estimators_[t].tree_
    pred_tree = tree.predict(xtr.astype(np.float32))
    apply_tree = tree.apply(xtr.astype(np.float32))
    # draw_tree(ensemble, t)
    # save_tree_graph(ensemble, t)
    return pred_tree

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

# def tiny_poisson(l):
#     mean_pred, ppf_obs, poi_mod = [None for i in range(3)]
#     xtr = np.array([item[1:] for item in l])
#     ytr = np.array([item[0] for item in l]).reshape(-1, 1)
#     try:
#         poi_mod = sm.Poisson(ytr, xtr).fit(method="newton", maxiter=500)
#         mean_pred = poi_mod.predict(xtr)  # or use a new x
#         sf_obs = stats.poisson.sf(2 - 1, mean_pred)  # average over x in sample
#         pmf_obs = stats.poisson.pmf(2, mean_pred)
#         ppf_obs = stats.poisson.ppf(q=0.95, mu=mean_pred)  # average over x in sample
#
#     except np.linalg.LinAlgError as e:
#         if 'Singular matrix' in str(e):
#             print("Ignored a singular matrix.", np.isnan(xtr).any(), np.isinf(xtr).any(), np.isneginf(xtr).any())
#             print("negative? ", xtr[xtr==0].any())
#     return [poi_mod, mean_pred, ppf_obs]
#
# def poisson_predictions_leaf_nodes(dicori):
#     dicpred = defaultdict(tuple)
#     voltes = 1
#     for key in sorted(dicori.keys()):
#         print()
#         print("Building tiny Poisson per leaf node")
#         print("Tree: {0}\t Leaf node: {1}\t Samples received: {2}\t ".format(key[0], key[1], len(dicori[key])))
#         poi_mod, mean_pred, prob_to_obs = tiny_poisson(dicori[key])
#         voltes -=1
#         if poi_mod != None:
#             dicpred[key] = (poi_mod, mean_pred, prob_to_obs)
#         else:
#             dicpred[key] = ("Singular matrix", )
#         # if voltes==0:
#         #     break
#     return dicpred

def ensemble_predictions_leaf_nodes(dicori):
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


def compare_models_plot(dicori, dicpoi, dicens):
    rows = 3
    cols = 5
    fig, ax = plt.subplots(rows, cols, sharey='row')
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    idx = list(itertools.product(range(rows), range(cols)))
    k = 0
    for key in sorted(dicori.keys()):
        l = dicori[key]
        ori_samples = np.array([item[0] for item in l]).reshape(-1, 1)
        ens_samples = dicens[key][0]
        if len(dicpoi[key])==1:
            poi_samples = []
        else:
            poi_samples = dicpoi[key][2]

        i = idx[k][0]
        j = idx[k][1]

        ax[i, j].grid(color='gray', linestyle='-', linewidth=1, zorder=0)
        ax[i, j].hist(ori_samples, color="darkblue", label="Original")
        ax[i, j].axvline(x=ens_samples[0], linewidth=3, color='orange', label="Ensemble")
        if len(poi_samples)>0:
            ax[i, j].hist(poi_samples, color="green", label="Poisson")

        title = "Tree: {0}, Node: {1}, Samples/Node: {2}".format(key[0], key[1], len(dicori[key]))
        ax[i, j].set_title(title, size=16)
        ax[i, j].legend()

        k += 1
    # ax[i, j].set_visible(False)
    plt.show()

def poisson_predictions_testing(dicori, dicpoi, xtest):
    p = []
    for key in sorted(dicori.keys()):
        if len(dicpoi[key]) > 1:
            model = dicpoi[key][0]
            mean = dicpoi[key][1]
            preds = dicpoi[key][2]
            ypred_mean = model.predict(xtest)  # or use a new x
            ypred_ppf = stats.poisson.ppf(q=0.85, mu=ypred_mean)  # average over x in sample
            ypred_ppf[ypred_ppf > 300] = 1
            p.append(ypred_ppf)
        else:
            print("Avoiding model w/ singular matrix")
    return np.array(p).T

def poisson_predictions_testing_nl(dicori, dicpoi, xtest):
    p = []
    for key in sorted(dicori.keys()):
        print(key)
        if len(dicpoi[key]) > 1:
            model = dicpoi[key][0]
            mean = dicpoi[key][1]
            preds = dicpoi[key][2]
            ypred_mean = model.predict(xtest)  # or use a new x
            ypred_ppf = stats.poisson.ppf(q=0.85, mu=ypred_mean)  # average over x in sample
            ypred_ppf[ypred_ppf > 300] = 1
            p.append(ypred_ppf)
    return np.array(p).T


def ensemble_predictions_testing(ensemble, xtest):
    ypred = ensemble.predict(xtest)
    return ypred

def plot_poisson_ensemble_raw(ypred_poi, ypred_ens, Yavg, ytest):
    print(ypred_poi.shape, ypred_ens.shape)

    plt.rcParams['axes.facecolor'] = 'lightgray'
    plt.rc('legend', **{'fontsize': 20})
    fig, ax = plt.subplots(1, 2)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    ax[0].grid(color='#7E888E', linestyle='-', linewidth=1, zorder=1)
    ax[0].hist(ypred_poi, bins=100, color="#204260", zorder=2, label="Poisson")
    ax[0].hist(ypred_ens, bins=40, color="#53A540", zorder=3, label="Ensemble")
    ax[0].axvline(x=Yavg, linewidth=8, color="#DBA521", zorder=0, label="Target mean")
    ax[0].legend().get_frame().set_facecolor("white")

    ax[1].grid(color='#7E888E', linestyle='-', linewidth=1, zorder=1)
    ax[1].hist(ypred_poi, bins=100, color="#204260", zorder=3, label="Poisson")
    ax[1].hist(ytest, bins=100, color="#9F5031", zorder=2, label="Raw target")
    ax[1].set_ylim(0, 600)
    ax[1].legend().get_frame().set_facecolor("white")

    plt.show()


def test_poi_nb_zip_zinb_raw_data(meta, m):
    Y = m[:, 0]
    X = m[:, 1:]
    Ynz, Xnz = trim_value(Y, X, 0)
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.60, random_state=77)

    print("Training with: ", xtrain.shape, ytrain.shape)
    print("Testing with: ", xtest.shape, ytest.shape)

    print()
    print("Model: Poisson")
    poi_mod = sm.Poisson(ytrain, xtrain).fit(method="newton", maxiter=50)
    poi_mean_pred = poi_mod.predict(xtest)
    poi_ppf_obs = stats.poisson.ppf(q=0.95, mu=poi_mean_pred)
    poi_rmse = np.sqrt(mean_squared_error(ytest, poi_ppf_obs))

    print("Model: Zero Inflated Poisson")
    zip_mod = sm.ZeroInflatedPoisson(ytrain, xtrain).fit(method="newton", maxiter=50)
    zip_mean_pred = zip_mod.predict(xtest, exog_infl=np.ones((len(xtest), 1)))
    zip_ppf_obs = stats.poisson.ppf(q=0.95, mu=zip_mean_pred)
    zip_rmse = np.sqrt(mean_squared_error(ytest, zip_ppf_obs))

    print("Model: Zero Inflated Neg. Binomial")
    zinb_mod = sm.ZeroInflatedNegativeBinomialP(ytrain, xtrain).fit(method="newton", maxiter=50)
    zinb_pred = zinb_mod.predict(xtest, exog_infl=np.ones((len(xtest), 1)))
    zinb_rmse = np.sqrt(mean_squared_error(ytest, zinb_pred))

    print()
    print("Model: Zero Inflated Neg. Binomial")
    zinb_mod = sm.ZeroInflatedNegativeBinomialP(ytrain, xtrain).fit(method="newton", maxiter=50)
    zinb_pred = zinb_mod.predict(xtest)
    zinb_rmse = np.sqrt(mean_squared_error(ytrain, zinb_pred))

    print("RMSE Poisson: ", poi_rmse)
    print("RMSE Negative Binomial: ", nb_rmse)
    print("RMSE Zero-Inflated Poisson", zip_rmse)
    print("RMSE Zero-Inflated Negative Binomial: ", zinb_rmse)



def test_poi_nb_zip_zinb_tiny_subset(meta, m):
    exog_names = r"rowid;latitude;longitude;target;dbuiltup;dforest;drecreation;dbrr;dwrl;dwrn;dwrr;dcamping;dcaravan;dcross;dgolf;dheem;dhaven;dsafari;dwater;attr;dbath;lu;lc;maxmeanhaz;maxstdhaz".split(";")[4:]

    np.random.seed(2)

    randint = np.random.randint(0, high=len(m)-1, size=800)

    msel = m[randint,:]

    Y = msel[:, 0]
    X = msel[:, 1:]

    # Ynz, Xnz = trim_value(Y, X, 0)

    print("Msel shape: ", msel.shape)

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.60, random_state=42)

    print(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape)

    print
    print("Model: Poisson")
    poi_mod = sm.Poisson(ytrain, xtrain).fit(method="newton", maxiter=50)
    poi_mean_pred = poi_mod.predict(xtest)
    poi_ppf_obs = stats.poisson.ppf(q=0.95, mu=poi_mean_pred)
    poi_rmse = np.sqrt(mean_squared_error(ytest, poi_ppf_obs))
    # print(np.unique(poi_ppf_obs, return_counts=True))
    print("RMSE Poisson: ", poi_rmse)
    # print(poi_mod.summary(yname='tickbites', xname=exog_names))

    print
    print("Model: Neg. Binomial")
    nb_mod = sm.NegativeBinomial(ytrain, xtrain).fit(start_params = None, method = 'newton', maxiter=50)
    nb_pred = nb_mod.predict(xtest)
    nb_rmse = np.sqrt(mean_squared_error(ytest, nb_pred))
    # print(np.unique(nb_pred, return_counts=True))
    print("RMSE Negative Binomial: ", nb_rmse)

    print
    print("Model: Zero Inflated Poisson")
    zip_mod = sm.ZeroInflatedPoisson(ytrain, xtrain).fit(method="newton", maxiter=50)
    zip_mean_pred = zip_mod.predict(xtest, exog_infl=np.ones((len(xtest), 1)))
    zip_ppf_obs = stats.poisson.ppf(q=0.95, mu=zip_mean_pred)
    zip_rmse = np.sqrt(mean_squared_error(ytest, zip_ppf_obs))
    print("RMSE Zero-Inflated Poisson", zip_rmse)

    print
    print("Model: Zero Inflated Neg. Binomial")
    zinb_mod = sm.ZeroInflatedNegativeBinomialP(ytrain, xtrain).fit(method="newton", maxiter=50)
    zinb_pred = zinb_mod.predict(xtest, exog_infl=np.ones((len(xtest), 1)))
    zinb_rmse = np.sqrt(mean_squared_error(ytest, zinb_pred))
    print("RMSE Zero-Inflated Negative Binomial: ", zinb_rmse)







    # fig, ax = plt.subplots(2, 4)
    # plt.subplots_adjust(wspace=0.5, hspace=0.5)
    #
    # ax[0,0].set_title("Poisson vs. Raw data", size=18)
    # ax[0,0].grid(color='gray', linestyle='-', linewidth=1, zorder=1)
    # ax[0,0].hist(ytest, bins=50, color="#204260", zorder=2, label="Samples")
    # ax[0,0].hist(poi_ppf_obs, bins=50, color="orange", zorder=3, label="Poisson")
    #
    # ax[0,1].set_title("Negative Binomial vs. Raw data", size=18)
    # ax[0,1].grid(color='gray', linestyle='-', linewidth=1, zorder=1)
    # ax[0,1].hist(ytest, bins=50, color="#204260", zorder=2, label="Samples")
    # ax[0,1].hist(nb_pred, bins=50, color="orange", zorder=3, label="Neg. Bin.")
    #
    # ax[0,2].set_title("Zero-inflated Poisson vs. Raw data", size=18)
    # ax[0,2].grid(color='gray', linestyle='-', linewidth=1, zorder=1)
    # ax[0,2].hist(ytest, bins=50, color="#204260", zorder=2, label="Samples")
    # ax[0,2].hist(zip_ppf_obs, bins=50, color="orange", zorder=3, label="ZIP")
    #
    # ax[0,3].set_title("Zero-inflated Neg. Bin. vs. Raw data", size=18)
    # ax[0,3].grid(color='gray', linestyle='-', linewidth=1, zorder=1)
    # ax[0,3].hist(ytest, bins=50, color="#204260", zorder=2, label="Samples")
    # ax[0,3].hist(zinb_pred, bins=50, color="orange", zorder=3, label="ZINB")
    #
    # plt.show()

def tiny_poisson(l):
    print("\t\tRunning Poisson")
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
            print("\t\t\tIgnored a singular matrix.")
    except ValueError:
        print("\t\t\tIgnored output containing np.nan or np.inf")

    return [poi_mod, poi_ppf_obs, poi_rmse]


def tiny_negbin(l):
    print("\t\tRunning NegBin")
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
            print("\t\t\tIgnored a singular matrix.")
    except ValueError:
        print("\t\t\tIgnored output containing np.nan or np.inf")

    return [nb_mod, nb_pred, nb_rmse]


def tiny_zip(l):
    print("\t\tRunning Zero-Inflated Poisson")
    zip_mod, zip_ppf_obs, zip_pred = [None for i in range(3)]
    zip_rmse = 0
    xtr = np.array([item[1:] for item in l])
    ytr = np.array([item[0] for item in l]).reshape(-1, 1)
    try:
        zip_mod = sm.ZeroInflatedPoisson(ytr, xtr).fit(method="newton", maxiter=50)
        zip_mean_pred = zip_mod.predict(xtr, exog_infl=np.ones((len(xtr), 1)))
        zip_ppf_obs = stats.poisson.ppf(q=0.95, mu=zip_mean_pred)
        zip_rmse = np.sqrt(mean_squared_error(ytr, zip_ppf_obs))
    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            print("\t\t\tIgnored a singular matrix.")
    except ValueError:
        print("\t\t\tIgnored output containing np.nan or np.inf")
    return [zip_mod, zip_ppf_obs, zip_rmse]

def tiny_zinb(l):
    print("\t\tRunning Zero-Inflated NegBin")
    zinb_mod, zinb_pred = [None for i in range(2)]
    zinb_rmse = 0

    xtr = np.array([item[1:] for item in l])
    ytr = np.array([item[0] for item in l]).reshape(-1, 1)

    try:
        zinb_mod = sm.ZeroInflatedNegativeBinomialP(ytr, xtr).fit(method="newton", maxiter=50)
        zinb_pred = zinb_mod.predict(xtr, exog_infl=np.ones((len(xtr), 1)))
        zinb_rmse = np.sqrt(mean_squared_error(ytr, zinb_pred))

    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            print("\t\t\tIgnored a singular matrix.")
    except ValueError:
        print("\t\t\tIgnored output containing np.nan or np.inf")
    return [zinb_mod, zinb_pred, zinb_rmse]


def process_pack(key, dic, mod, pred, rmse):
    if mod != None:
        dic[key] = (mod, pred, rmse)
    else:
        dic[key] = ("Singular matrix",)
    return dic


def fitting_four_models_leaf_nodes(dicori):
    poi_dic, nb_dic, zip_dic, zinb_dic = [{} for i in range(4)]
    for key in sorted(dicori.keys()):
        print()
        print("Tree: {0}\t Leaf node: {1}\t Samples received: {2}\t ".format(key[0], key[1], len(dicori[key])))
        poi_mod, poi_pred, poi_rmse = tiny_poisson(dicori[key])
        nb_mod, nb_pred, nb_rmse = tiny_negbin(dicori[key])
        zip_mod, zip_pred, zip_rmse = tiny_zip(dicori[key])
        zinb_mod, zinb_pred, zinb_rmse = tiny_zinb(dicori[key])

        poi_dic = process_pack(key, poi_dic, poi_mod, poi_pred, poi_rmse)
        nb_dic = process_pack(key, nb_dic, nb_mod, nb_pred, nb_rmse)
        zip_dic = process_pack(key, zip_dic, zip_mod, zip_pred, zip_rmse)
        zinb_dic = process_pack(key, zinb_dic, zinb_mod, zinb_pred, zinb_rmse)

    return [poi_dic, nb_dic, zip_dic, zinb_dic]


def predicting_four_models_leaf_nodes(xtest, pack):
    l = []
    for dictionary in pack[2:3]:
        for key in sorted(dictionary.keys()):
            print("Key: ", key)
            if len(dictionary[key]) == 3:
                model = dictionary[key][0]
                predtr = dictionary[key][1]
                rmsetr = dictionary[key][2]
                print("Type: ", type(model))
                print(model.summary())
                predte = model.predict(xtest)
                l.append(predte)

            plt.hist(predte, bins=100)
            plt.show()

        mean_pred  = np.mean(np.array(l), axis=0)
        print(np.isnan(mean_pred).any(), np.isinf(mean_pred).any(), np.isneginf(mean_pred).any())
        print(np.unique(mean_pred, return_counts=True))

        break

    return []

def test_varying_samples_per_node(meta_m, m):
    print("Type m: ", m.dtype)

    Y = m[:,0]
    X = m[:,1:]

    # Ynz, Xnz = trim_value(Y, X, 0)

    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.60, random_state=0)

    print('Type xtrain: ', xtrain.dtype)
    print("Raw data: ", Y.shape, X.shape)
    print("Train data", ytrain.shape, xtrain.shape)
    print("Test data: ", ytest.shape, xtest.shape)

    for samples_per_leaf in range(500, 600, 100):

        print("Samples per leaf node: ", samples_per_leaf)

        ensemble = RandomForestRegressor(n_estimators=1, min_samples_leaf=samples_per_leaf, bootstrap=False)

        ensemble.fit(xtrain, ytrain)

        leaves = ensemble.apply(xtrain)

        print(leaves)

        print(leaves.shape)

        dicori = samples_per_leaf_node(leaves, xtrain, ytrain)

        pack = fitting_four_models_leaf_nodes(dicori)

        pred_ = predicting_four_models_leaf_nodes(xtest, pack)

        # dicens = ensemble_predictions_leaf_nodes(dicori)

        break

    # l, li = [[] for i in range(2)]
    # pack = [poi_dic, nb_dic, zip_dic, zinb_dic]
    # for item in pack:
    #     for key in sorted(item.keys()):
    #         li.append(len(item[key]))
    #         if len(item[key]) == 3:
    #             print(item[key])
    #     l.append(li)
    #     li = []
    #
    # # IRENE RECORDA CHECAR PER A NANS
    # plt.subplot(2, 2, 1)
    # plt.hist(l[0], bins=50)
    # plt.subplot(2, 2, 2)
    # plt.hist(l[1], bins=50)
    # plt.subplot(2, 2, 3)
    # plt.hist(l[2], bins=50)
    # plt.subplot(2, 2, 4)
    # plt.hist(l[3], bins=50)
    #
    # plt.show()



def test_poisson_varying_number_of_zeros():
    pass


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

# test_poi_nb_zip_zinb_raw_data(meta_m, m)
# test_poi_nb_zip_zinb_tiny_subset(meta_m, m)
test_varying_samples_per_node(meta_m, m)




#
# print("Type m: ", m.dtype)
#
# Y = m[:,0]
# X = m[:,1:]
#
# Ynz, Xnz = trim_value(Y, X, 0)
#
# Yavg = np.mean(Ynz)
#
# xtrain, xtest, ytrain, ytest = train_test_split(Xnz, Ynz, train_size=0.60, random_state=0)
#
# print('Type xtrain: ', xtrain.dtype)
# print("Raw data: ", Y.shape, X.shape)
# print("Trim data: ", Ynz.shape, Xnz.shape)
# print("Train data", ytrain.shape, xtrain.shape)
# print("Test data: ", ytest.shape, xtest.shape)
#
# ensemble = RandomForestRegressor(n_estimators=100, min_samples_leaf=500, bootstrap=False)
#
# ensemble.fit(xtrain, ytrain)
#
# leaves = ensemble.apply(xtrain)
#
# dicori = samples_per_leaf_node(leaves, xtrain, ytrain)

# dicpoi = poisson_predictions_leaf_nodes(dicori)
#
# dicens = ensemble_predictions_leaf_nodes(dicori)

# ypred_poi = np.mean(poisson_predictions_testing(dicori, dicpoi, xtest), axis=1)
#
# ypred_ens = ensemble_predictions_testing(ensemble, xtest)
#
# plot_poisson_ensemble_raw(ypred_poi, ypred_ens, Yavg, ytest)




