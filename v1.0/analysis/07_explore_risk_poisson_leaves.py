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
import seaborn as sb


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

def tiny_poisson(l):
    mean_pred, ppf_obs, poi_mod = [None for i in range(3)]
    xtr = np.array([item[1:] for item in l])
    ytr = np.array([item[0] for item in l]).reshape(-1, 1)
    try:
        poi_mod = sm.Poisson(ytr, xtr).fit(method="newton", maxiter=500)
        mean_pred = poi_mod.predict(xtr)  # or use a new x
        sf_obs = stats.poisson.sf(2 - 1, mean_pred)  # average over x in sample
        pmf_obs = stats.poisson.pmf(2, mean_pred)
        ppf_obs = stats.poisson.ppf(q=0.95, mu=mean_pred)  # average over x in sample

    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            print("Ignored a singular matrix.", np.isnan(xtr).any(), np.isinf(xtr).any(), np.isneginf(xtr).any())
            print("negative? ", xtr[xtr==0].any())
    return [poi_mod, mean_pred, ppf_obs]

def poisson_predictions_leaf_nodes(dicori):
    dicpred = defaultdict(tuple)
    voltes = 1
    for key in sorted(dicori.keys()):
        print()
        print("Building tiny Poisson per leaf node")
        print("Tree: {0}\t Leaf node: {1}\t Samples received: {2}\t ".format(key[0], key[1], len(dicori[key])))
        poi_mod, mean_pred, prob_to_obs = tiny_poisson(dicori[key])
        voltes -=1
        if poi_mod != None:
            dicpred[key] = (poi_mod, mean_pred, prob_to_obs)
        else:
            dicpred[key] = ("Singular matrix", )
        # if voltes==0:
        #     break
    return dicpred

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

################
# Main program #
################

path_in = r"/media/irene/MyPassport/acies/workspace/Special/IJHG/data/versions/country_features/v12/nl_risk_features_v1.12.csv"

metacols = range(0, 3)
datacols = range(3, 25)

meta_m = np.loadtxt(path_in, delimiter=";", usecols=metacols, skiprows=1)
m = np.loadtxt(path_in, delimiter=";", usecols=datacols, skiprows=1, dtype=np.float32)
print("Type m: ", m.dtype)

Y = m[:,0]
X = m[:,1:]

Ynz, Xnz = trim_value(Y, X, 0)

xtrain, xtest, ytrain, ytest = train_test_split(Xnz, Ynz, train_size=0.60, random_state=0)

print('Type xtrain: ', xtrain.dtype)

print("Raw data: ", Y.shape, X.shape)
print("Trim data: ", Ynz.shape, Xnz.shape)
print("Train data", ytrain.shape, xtrain.shape)
print("Test data: ", ytest.shape, xtest.shape)

ensemble = RandomForestRegressor(n_estimators=1, min_samples_leaf=300, bootstrap=False)

ensemble.fit(xtrain, ytrain)

leaves = ensemble.apply(xtrain)

dicori = samples_per_leaf_node(leaves, xtrain, ytrain)

dicpoi = poisson_predictions_leaf_nodes(dicori)

dicens = ensemble_predictions_leaf_nodes(dicori)

save_tree_graph(ensemble, 0)

compare_models_plot(dicori, dicpoi, dicens)

